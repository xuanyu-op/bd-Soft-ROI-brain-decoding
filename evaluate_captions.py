import os
import pickle
import json
import argparse
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from contextlib import nullcontext

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

os.environ["TRANSFORMERS_OFFLINE"] = "1"


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"===== {text.center(68)} =====")
    print("=" * 80)


class CaptionEvaluator:
    """Encapsulates all caption evaluation logic."""

    def __init__(self, clip_model_path, device, clip_image_bs=32, clip_text_bs=32):
        self.device = device
        self.clip_image_bs = clip_image_bs
        self.clip_text_bs = clip_text_bs
        self.autocast_context = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        print("Loading CLIP model and processor for evaluation...")
        self.clip_model = CLIPModel.from_pretrained(
            clip_model_path, local_files_only=True
        ).to(self.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_model_path, local_files_only=True
        )
        self.ref_text_embedding_cache = {}
        print("CLIP assets loaded successfully.")

    def calculate_nlg_metrics(self, generated_captions, reference_captions_list):
        """Compute BLEU, ROUGE-L, and CIDEr metrics."""
        print_header("Calculating NLG Metrics (BLEU, ROUGE-L, CIDEr)")

        hyps = {
            str(i): [{"caption": str(caption)}]
            for i, caption in enumerate(generated_captions)
        }
        refs = {
            str(i): [{"caption": str(r)} for r in ref_list]
            for i, ref_list in enumerate(reference_captions_list)
        }

        tokenizer = PTBTokenizer()
        hyps_tokenized = tokenizer.tokenize(hyps)
        refs_tokenized = tokenizer.tokenize(refs)

        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        results = {}
        for scorer, method in tqdm(scorers, desc="Computing NLG Scores"):
            score, _ = scorer.compute_score(refs_tokenized, hyps_tokenized)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    results[m] = s
            else:
                results[method] = score

        if results.get("CIDEr", 0.0) > 5.0:
            results["CIDEr"] /= 10.0
        results["CIDEr"] = max(0.0, min(1.0, results.get("CIDEr", 0.0)))

        return results

    def calculate_clip_s(self, image_paths, captions):
        """Compute CLIP Score, i.e., image-text similarity."""
        print_header("Calculating CLIP Score")

        tokenizer = self.clip_processor.tokenizer

        valid_images, valid_captions = [], []
        for img_path, cap in zip(image_paths, captions):
            if not (cap and str(cap).strip()):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
                valid_captions.append(str(cap))
            except Exception as e:
                warnings.warn(f"Skipping image {img_path} due to error: {e}")
                continue

        if not valid_captions:
            warnings.warn("No valid image-caption pairs found for CLIP scoring.")
            return 0.0

        max_length = tokenizer.model_max_length
        content_length = max_length - 2
        text_chunks_for_processing = []
        caption_to_chunks_map = []

        # Split overly long captions into multiple chunks for separate encoding
        for caption in valid_captions:
            tokens = tokenizer(
                caption, add_special_tokens=False, truncation=False
            ).input_ids
            current_caption_chunks_indices = []
            chunks_of_tokens = (
                [tokens[j:j + content_length] for j in range(0, len(tokens), content_length)]
                if tokens
                else [[]]
            )

            for chunk_tokens in chunks_of_tokens:
                chunk_text = tokenizer.decode(chunk_tokens)
                text_chunks_for_processing.append(chunk_text)
                current_caption_chunks_indices.append(
                    len(text_chunks_for_processing) - 1
                )
            caption_to_chunks_map.append(current_caption_chunks_indices)

        if not text_chunks_for_processing:
            return 0.0

        # Encode text and image features in batches, then compute image-text similarity
        with torch.no_grad(), self.autocast_context:
            all_text_features = []
            for i in tqdm(
                range(0, len(text_chunks_for_processing), self.clip_text_bs),
                desc="Encoding text chunks",
            ):
                text_batch = text_chunks_for_processing[i:i + self.clip_text_bs]
                with (
                    torch.cuda.amp.autocast(enabled=False)
                    if self.device.type == "cuda"
                    else nullcontext()
                ):
                    text_inputs = self.clip_processor(
                        text=text_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(self.device)

                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
                all_text_features.append(text_features)
            all_text_features = torch.cat(all_text_features)

            caption_scores = []
            for i in tqdm(
                range(0, len(valid_images), self.clip_image_bs),
                desc="Comparing images and texts",
            ):
                image_batch = valid_images[i:i + self.clip_image_bs]
                image_inputs = self.clip_processor(
                    images=image_batch, return_tensors="pt", padding=True
                ).to(self.device)
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                for j in range(image_features.shape[0]):
                    caption_idx = i + j
                    chunk_indices = caption_to_chunks_map[caption_idx]
                    if not chunk_indices:
                        caption_scores.append(-1.0)
                        continue

                    caption_text_features = all_text_features[chunk_indices]
                    current_image_feature = image_features[j].unsqueeze(0)
                    sims = (current_image_feature @ caption_text_features.T).squeeze()
                    aggregated_sim = (
                        torch.max(sims).item() if sims.ndim > 0 else sims.item()
                    )
                    caption_scores.append(aggregated_sim)

        final_score = np.mean([(s + 1) / 2 for s in caption_scores if s is not None])
        return final_score

    def _get_text_embeddings(self, text_list):
        """Get text embeddings with caching to avoid repeated encoding."""
        embeddings, texts_to_process, indices_to_process = [], [], []
        for i, text in enumerate(text_list):
            if text in self.ref_text_embedding_cache:
                embeddings.append(self.ref_text_embedding_cache[text])
            else:
                embeddings.append(None)
                texts_to_process.append(text)
                indices_to_process.append(i)

        if texts_to_process:
            inputs = self.clip_processor(
                text=texts_to_process,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad(), self.autocast_context:
                new_embeds = self.clip_model.get_text_features(**inputs)
            for i, embed in zip(indices_to_process, new_embeds):
                embeddings[i] = embed
                self.ref_text_embedding_cache[text_list[i]] = embed

        return torch.stack(embeddings)

    def calculate_refclip_s(self, captions, refs_list):
        """Compute RefCLIP Score, i.e., semantic similarity between generated and reference texts."""
        print_header("Calculating RefCLIP Score")
        scores = []
        for gen_caption, refs in tqdm(
            zip(captions, refs_list),
            total=len(captions),
            desc="Calculating RefCLIP",
        ):
            if not gen_caption or not refs:
                scores.append(0.0)
                continue

            with self.autocast_context:
                gen_embed = self._get_text_embeddings([gen_caption])
                ref_embeds = self._get_text_embeddings(refs)
                gen_embed_norm = gen_embed / gen_embed.norm(dim=-1, keepdim=True)
                ref_embeds_norm = ref_embeds / ref_embeds.norm(dim=-1, keepdim=True)
                sims = (gen_embed_norm @ ref_embeds_norm.T).squeeze()
                avg_sim = sims.mean().item()
                scores.append((avg_sim + 1) / 2)

        return np.mean(scores) if scores else 0.0


def main(args):
    """Main function: load data, align samples, run evaluation, and print results."""
    print_header("Starting Caption Evaluation")

    print("Loading generated and reference data...")
    try:
        with open(args.generated_captions_pkl, "rb") as f:
            generated_data = pickle.load(f)
        with open(args.coco_captions_path, "r") as f:
            reference_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return

    print("Aligning generated captions with reference data...")
    generated_captions = []
    reference_captions_list = []
    image_paths = []

    # Use a set to record valid image IDs and avoid duplicates while speeding up lookup
    valid_image_ids_for_clip = set()
    for image_id, gen_cap in generated_data.items():
        image_id_str = str(image_id)
        if image_id_str in reference_data:
            generated_captions.append(gen_cap if gen_cap else "")
            reference_captions_list.append(reference_data[image_id_str])

            # Search for the corresponding image file for CLIP Score
            found_image = False
            for ext in [".jpg", ".png", ".jpeg"]:
                image_path = os.path.join(args.image_dir, f"{image_id_str}{ext}")
                if os.path.exists(image_path):
                    image_paths.append(image_path)
                    valid_image_ids_for_clip.add(image_id_str)
                    found_image = True
                    break

            # If the image does not exist, append a placeholder and skip it later during CLIP scoring
            if not found_image:
                image_paths.append(None)

    num_samples = len(generated_captions)
    if num_samples == 0:
        print("Error: No matching samples found between the .pkl file and reference JSON. Aborting.")
        return

    print(f"Found {num_samples} matching samples for NLG evaluation.")
    print(f"Found {len(valid_image_ids_for_clip)} matching images for CLIP Score evaluation.")

    print("Initializing evaluator...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    evaluator = CaptionEvaluator(
        clip_model_path=args.clip_model_path,
        device=device,
        clip_image_bs=args.clip_image_bs,
        clip_text_bs=args.clip_text_bs,
    )

    all_scores = {}

    # Compute conventional text generation metrics
    nlg_scores = evaluator.calculate_nlg_metrics(
        generated_captions, reference_captions_list
    )
    all_scores.update(nlg_scores)

    # Keep only samples with available images for CLIP Score computation
    clip_image_paths = [p for p in image_paths if p is not None]
    clip_captions = [
        cap for cap, p in zip(generated_captions, image_paths) if p is not None
    ]

    # Compute image-text similarity score
    if clip_image_paths:
        clip_score = evaluator.calculate_clip_s(clip_image_paths, clip_captions)
        all_scores["CLIP_Score"] = clip_score
    else:
        all_scores["CLIP_Score"] = 0.0
        warnings.warn("No images found, CLIP Score set to 0.0")

    # Compute semantic similarity between generated texts and reference texts
    refclip_score = evaluator.calculate_refclip_s(
        generated_captions, reference_captions_list
    )
    all_scores["RefCLIP_Score"] = refclip_score

    print_header("Final Evaluation Results")

    # Use pandas to print the final results in a table format
    df = pd.DataFrame(list(all_scores.items()), columns=["Metric", "Score"])
    df["Score"] = df["Score"].apply(lambda x: f"{x:.4f}")
    print(df.to_string(index=False))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate generated image captions from a .pkl file."
    )

    parser.add_argument(
        "--generated_captions_pkl",
        type=str,
        required=True,
        help="Path to the input .pkl file containing generated captions (key: image_id, value: caption).",
    )
    parser.add_argument(
        "--coco_captions_path",
        type=str,
        required=True,
        help="Path to the JSON file with reference captions (e.g., coco_id_to_captions.json).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing the original COCO images, needed for CLIP Score.",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        required=True,
        help="Path to the local directory of the CLIP model (e.g., 'openai/clip-vit-large-patch14').",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for evaluation ('cuda:0' or 'cpu').",
    )
    parser.add_argument(
        "--clip_image_bs",
        type=int,
        default=32,
        help="Batch size for images in CLIP scoring.",
    )
    parser.add_argument(
        "--clip_text_bs",
        type=int,
        default=32,
        help="Batch size for text chunks in CLIP scoring.",
    )

    args = parser.parse_args()
    main(args)