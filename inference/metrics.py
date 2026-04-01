import warnings
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
from contextlib import nullcontext
from patch_evalcap import silence_evalcap


def calculate_clip_s(images, captions, clip_model, processor, device, batch_size_img=32, batch_size_txt=32):
    """
    Calculate CLIP-S between images and generated captions.
    """
    valid_images, valid_captions = [], []
    to_pil = ToPILImage()

    for img, cap in zip(images, captions):
        if not (cap and str(cap).strip()):
            continue
        std_img = None
        try:
            if isinstance(img, Image.Image):
                std_img = img.convert("RGB")
            elif isinstance(img, str):
                std_img = Image.open(img).convert("RGB")
            elif isinstance(img, torch.Tensor):
                if img.ndim == 4:
                    img = img.squeeze(0)
                if img.dtype == torch.uint8:
                    std_img = to_pil(img).convert("RGB")
                elif img.dtype in [torch.float16, torch.float32]:
                    if img.max() > 1.0:
                        img = img / 255.0
                    std_img = to_pil(img).convert("RGB")
            if std_img:
                valid_images.append(std_img)
                valid_captions.append(str(cap))
        except Exception:
            continue

    if not valid_captions:
        return 0.0
    
    # Split long text into multiple chunks for subsequent encoding.
    tokenizer = processor.tokenizer
    max_length = tokenizer.model_max_length
    content_length = max_length - 2

    text_chunks_for_processing = []
    caption_to_chunks_map = []

    #Encode the text in batches.
    for caption in valid_captions:
        tokens = tokenizer(caption, add_special_tokens=False, truncation=False).input_ids
        if not tokens:
            chunks = [[]]
        else:
            chunks = [tokens[j:j + content_length] for j in range(0, len(tokens), content_length)]

        current_indices = []
        for ch in chunks:
            text_chunks_for_processing.append(tokenizer.decode(ch))
            current_indices.append(len(text_chunks_for_processing) - 1)
        caption_to_chunks_map.append(current_indices)

    autocast_context = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()

    with torch.no_grad(), autocast_context:
        all_text_features = []
        for i in range(0, len(text_chunks_for_processing), batch_size_txt):
            batch = text_chunks_for_processing[i:i + batch_size_txt]
            # Disable autocast during text preprocessing if needed for numerical stability.
            with torch.cuda.amp.autocast(enabled=False) if device.type == 'cuda' else nullcontext():
                inputs = processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_text_features.append(feats)

        if not all_text_features:
            return 0.0
        all_text_features = torch.cat(all_text_features)

        #Encode images in batches and compute similarity with the corresponding text features.
        caption_scores = []
        for i in range(0, len(valid_images), batch_size_img):
            batch = valid_images[i:i + batch_size_img]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            im_feats = clip_model.get_image_features(**inputs)
            im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)

            for j in range(im_feats.shape[0]):
                cap_idx = i + j
                chunk_indices = caption_to_chunks_map[cap_idx]
                if not chunk_indices:
                    caption_scores.append(-1.0)
                    continue

                txt_feats = all_text_features[chunk_indices]
                sims = (im_feats[j].unsqueeze(0) @ txt_feats.T).squeeze()
                agg_sim = sims.item() if sims.ndim == 0 else torch.max(sims).item()
                caption_scores.append(agg_sim)

    if not caption_scores:
        return 0.0
    return np.mean([(s + 1) / 2 for s in caption_scores])


def _get_text_embeddings(text_list, clip_model, processor, device, cache):
    # Compute text features for RefCLIP and use caching to avoid redundant encoding.
    embeddings, texts_to_process, indices_to_process = [], [], []
    for i, text in enumerate(text_list):
        if text in cache:
            embeddings.append(cache[text])
        else:
            embeddings.append(None)
            texts_to_process.append(text)
            indices_to_process.append(i)

    if texts_to_process:
        with torch.no_grad():
            inputs = processor(text=texts_to_process, return_tensors="pt", padding=True, truncation=True).to(device)
            # Use autocast when generating text embeddings for RefCLIP.
            autocast_context = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
            with autocast_context:
                new_embeds = clip_model.get_text_features(**inputs)

            for i, embed in zip(indices_to_process, new_embeds):
                embeddings[i] = embed
                cache[text_list[i]] = embed

    if not embeddings:
        return torch.tensor([], device=device)
    return torch.stack(embeddings)


def calculate_refclip_s(captions, refs_list, clip_model, processor, device, cache):
    # Compute RefCLIP-S between generated captions and reference captions.
    scores = []
    autocast_context = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()

    for gen_caption, refs in zip(captions, refs_list):
        if not gen_caption or not refs:
            scores.append(0.0)
            continue

        with autocast_context:
            gen_embed = _get_text_embeddings([gen_caption], clip_model, processor, device, cache)
            ref_embeds = _get_text_embeddings(refs, clip_model, processor, device, cache)

            if gen_embed.numel() == 0 or ref_embeds.numel() == 0:
                scores.append(0.0)
                continue

            gen_embed_norm = gen_embed / gen_embed.norm(dim=-1, keepdim=True)
            ref_embeds_norm = ref_embeds / ref_embeds.norm(dim=-1, keepdim=True)

            sims = (gen_embed_norm @ ref_embeds_norm.T).squeeze()
            avg_sim = sims.mean().item()
            scores.append((avg_sim + 1) / 2)

    return np.mean(scores) if scores else 0.0


def calculate_nlg_metrics(captions, refs_list, spice_enabled=False):
    # Compute standard text generation metrics, including BLEU, CIDEr, and optional SPICE.
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider

    tokenizer = PTBTokenizer()
    num_samples = len(captions)

    hyps_dict = {
        str(i): [{"caption": str(captions[i]).lower().strip() if captions[i] else ""}]
        for i in range(num_samples)
    }
    refs_dict = {
        str(i): [{"caption": str(r).lower().strip()} for r in (refs_list[i] or [""])]
        for i in range(num_samples)
    }

    with silence_evalcap():
        hyps_tokenized = tokenizer.tokenize(hyps_dict)
        refs_tokenized = tokenizer.tokenize(refs_dict)

    scorers = [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]), (Cider(), "CIDEr")]

    if spice_enabled:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(), "SPICE"))

    results = {}
    for scorer, method in scorers:
        with silence_evalcap():
            score, _ = scorer.compute_score(refs_tokenized, hyps_tokenized)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results[m] = s
        else:
            results[method] = score

    return {
        'Bleu_1': results.get('Bleu_1', 0.0),
        'Bleu_2': results.get('Bleu_2', 0.0),
        'Bleu_3': results.get('Bleu_3', 0.0),
        'Bleu_4': results.get('Bleu_4', 0.0),
        'CIDEr': results.get('CIDEr', 0.0),
        'SPICE': results.get('SPICE', 0.0)
    }


def rerank_candidates_by_bleu4(candidate_captions, refs_list):
    # Rerank multiple candidate captions by BLEU-4 and return the best caption along with its BLEU scores.
    if not candidate_captions:
        return "", [-1.0] * 4

    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu

    bleu_scorer = Bleu(4)
    ptb_tokenizer = PTBTokenizer()

    refs_dict = {'0': [{'caption': r.lower().strip()} for r in refs_list]}
    with silence_evalcap():
        refs_tokenized = ptb_tokenizer.tokenize(refs_dict)

    best_caption = ""
    best_bleu_scores = [-1.0] * 4

    for candidate in candidate_captions:
        hyps_dict = {'0': [{'caption': candidate}]}
        with silence_evalcap():
            hyps_tokenized = ptb_tokenizer.tokenize(hyps_dict)
            bleu_scores, _ = bleu_scorer.compute_score(refs_tokenized, hyps_tokenized)

        if bleu_scores[3] > best_bleu_scores[3]:
            best_bleu_scores = bleu_scores
            best_caption = candidate

    return best_caption, best_bleu_scores