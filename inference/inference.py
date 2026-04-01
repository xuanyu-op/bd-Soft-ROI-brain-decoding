import os
import sys

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

import gc
import json
import torch
import argparse
import pickle
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM, CLIPModel, CLIPProcessor
from contextlib import nullcontext

import patch_evalcap
import common_utils
import models_fmri
import projector
import data_loader
import metrics
from prompt_optimizer import PromptOptimizer


def main():
    patch_evalcap.apply_patches()

    parser = argparse.ArgumentParser(description="fMRI to Text Inference")

    # Base inference settings
    parser.add_argument('--shikra_path', required=True)
    parser.add_argument('--brainroi_path', required=True)
    parser.add_argument('--clip_model_path', type=str, required=True)
    parser.add_argument('--adapter_path', default='model_weights/mm_projector.bin')
    parser.add_argument('--feat_dim', type=int, default=1024, choices=[1024, 4096])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_pkl', type=str, default='results.pkl')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--subj', type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_latents', type=int, default=256)

    parser.add_argument(
        '--fmri_encoder',
        type=str,
        default='brainroi',
        choices=['brainroi', 'brainrois'],
        help="Type of fMRI encoder (kept for CLI compatibility)"
    )

    # NIA/Soft-ROI related settings
    parser.add_argument('--softroi_root', type=str, required=True)
    parser.add_argument('--roi_root', type=str, required=True)
    parser.add_argument('--atlas_names', type=lambda s: [item.strip() for item in s.split(',')], required=True)
    parser.add_argument('--coord_norm', type=str, default='unit')
    parser.add_argument('--fusion_mode', type=str, default='concat')
    parser.add_argument('--gate_voxel_proj_dim', type=int, default=64)
    parser.add_argument('--attn_scale', type=str, default='sqrt')
    parser.add_argument('--attn_norm', type=str, default='layernorm')
    parser.add_argument('--attn_tau_init', type=float, default=1.0)
    parser.add_argument('--attn_tau_learnable', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--ffn_dropout', type=float, default=0.0)

    # Prompt optimization settings
    parser.add_argument('--prompt_opt_iters', type=int, default=0)
    parser.add_argument('--optimizer_model_path', type=str, default=None)
    parser.add_argument('--eval_image_dir', type=str, default=None)
    parser.add_argument('--stageA_precision', type=str, default='4bit', choices=['4bit', '8bit', 'fp16', 'bf16'])
    parser.add_argument('--rank_metric', type=str, default='clip',
                        choices=['clip', 'refclip', 'align', 'bleu1', 'bleu4', 'cider', 'spice', 'match'])
    parser.add_argument('--enable_spice', action='store_true')
    parser.add_argument('--coco_captions_path', type=str, default='coco_id_to_captions.json')
    parser.add_argument('--prompt_init', type=str, default=None)
    parser.add_argument('--prompt_out', type=str, default=None,
                        help="File path to save the best optimized prompt string (NOT a directory)")

    parser.add_argument('--optimizer_max_new_tokens', type=int, default=256)
    parser.add_argument('--optimizer_temperature', type=float, default=0.8)
    parser.add_argument('--optimizer_top_p', type=float, default=0.95)
    parser.add_argument('--num_new_prompts_per_iter', type=int, default=2)
    parser.add_argument('--eval_chunk_size', type=int, default=500)
    parser.add_argument('--clip_image_bs', type=int, default=32)
    parser.add_argument('--clip_text_bs', type=int, default=32)
    parser.add_argument('--prompt_pool_size', type=int, default=3)

    # Text generation settings
    parser.add_argument('--do_sample', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--length_penalty', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)

    args = parser.parse_args()

    # Check whether the input prompt contains the image placeholder
    if '<image>' not in args.prompt:
        raise ValueError("Initial prompt must contain the '<image>' placeholder.")

    # If an evaluation image directory is specified, verify that it exists
    if args.eval_image_dir and not os.path.exists(args.eval_image_dir):
        raise FileNotFoundError(f"eval_image_dir does not exist at specified path: {args.eval_image_dir}")

    # Limit the maximum number of prompt optimization iterations
    args.prompt_opt_iters = min(args.prompt_opt_iters, 10)

    # Set random seeds and select the execution device
    common_utils.seed_everything(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    autocast_context = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    print(f"Using device: {device}")

    # If prompt optimization is enabled, search for the best prompt first
    best_prompt = args.prompt
    if args.prompt_opt_iters > 0:
        if not args.optimizer_model_path:
            raise ValueError("optimizer_model_path required")

        with open(args.coco_captions_path, 'r') as f:
            coco_id_to_captions = json.load(f)

        light_cache = data_loader.build_light_cache_for_prompt_opt(
            args.data_path,
            args.subj,
            coco_id_to_captions
        )

        optimizer = PromptOptimizer(
            args,
            device,
            {
                'cached_features': light_cache,
                'coco_id_to_captions': coco_id_to_captions
            }
        )
        best_prompt = optimizer.run()

        # Release memory and GPU memory used during the optimization stage
        del optimizer, light_cache
        gc.collect()
        torch.cuda.empty_cache()

    # Start the final inference stage
    common_utils.print_header("Starting Final Inference")
    print(f"Using Prompt: {best_prompt}")

    # Build the test dataset and load the BrainROI encoder
    test_data = data_loader.make_test_dataset(args.data_path, args.subj)
    atlas_labels = models_fmri.build_atlas_labels(args.softroi_root, args.atlas_names)
    voxel2emb = models_fmri.build_brainroi_from_args(args, atlas_labels, device)

    # Load the Shikra tokenizer and language model
    tokenizer = LlamaTokenizer.from_pretrained(args.shikra_path, padding_side='left', local_files_only=True)
    llama_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = LlamaForCausalLM.from_pretrained(
        args.shikra_path,
        torch_dtype=llama_dtype,
        local_files_only=True
    ).to(device).eval()

    # If the feature dimension is 1024, load the mm_projector as well
    if args.feat_dim == 1024:
        mm_projector = torch.nn.Linear(1024, 4096)
        projector.load_mm_projector_weights(mm_projector, args.adapter_path)
        mm_projector = mm_projector.to(device).eval()

    # Extract fMRI features for all test samples
    emb_voxel_list, image_id_list = [], []
    with torch.no_grad(), autocast_context:
        for voxels, _, coco_ids in tqdm(test_data, desc="Extracting fMRI features"):
            voxels = common_utils.preprocess_voxels(voxels, device)
            subject_batch = [f"subject_{args.subj}"]
            emb_voxel = voxel2emb(voxels, subject=subject_batch)
            emb_voxel_list.append(emb_voxel)
            image_id_list.append(data_loader._to_scalar_id(coco_ids))

    # Concatenate all extracted features and optionally project them into the LLM hidden space
    image_features = torch.cat(emb_voxel_list, dim=0)
    if args.feat_dim == 1024:
        with autocast_context:
            image_features = mm_projector(image_features.to(torch.float32)).to(model.dtype)
    else:
        image_features = image_features.to(model.dtype)

    # Build the prompt template with image token placeholders
    system = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    )
    user_image = " <im_start>" + "<im_patch>" * args.num_latents + "<im_end> "
    user_prompt = best_prompt.replace('<image>', user_image)
    input_text = system + user_prompt + " ASSISTANT:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)

    # Locate the starting position of the image token block for later feature insertion
    im_start_id = tokenizer.convert_tokens_to_ids("<im_start>")
    pos = torch.where(input_ids[0] == im_start_id)[0]
    assert len(pos) > 0
    image_start_token_pos = pos[0].item()

    # Build generation settings and initialize the results dictionary
    gen_kwargs = common_utils.get_generation_kwargs(args, tokenizer)
    results = {}

    print("Generating...")
    if image_features.shape[0] > 0:
        print("\n--- [First 30 Model-Selected Captions] ---")

    # Generate captions for each test sample and directly keep the model-selected top-1 output
    for cur_image_idx in tqdm(range(image_features.shape[0])):
        coco_id = str(image_id_list[cur_image_idx])

        cur_feat = image_features[cur_image_idx].unsqueeze(0).to(device=inputs_embeds.device)
        final_embeds = torch.cat((
            inputs_embeds[0, :image_start_token_pos + 1],
            cur_feat[0],
            inputs_embeds[0, image_start_token_pos + args.num_latents + 1:]
        ), dim=0).unsqueeze(0)

        with torch.inference_mode(), autocast_context:
            attn_mask = torch.ones(final_embeds.shape[:-1], dtype=torch.long, device=device)
            output_ids = model.generate(
                inputs_embeds=final_embeds,
                attention_mask=attn_mask,
                **gen_kwargs
            )

        responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        best_cap = common_utils.clean_caption(responses[0]) if len(responses) > 0 else ""
        results[coco_id] = best_cap

        # Print the first 30 generated captions for quick inspection
        if cur_image_idx < 30:
            print(
                f"  [{cur_image_idx + 1:02d}/30] ID {coco_id}: "
                f"{best_cap} (len: {len(best_cap.split())})"
            )
            if cur_image_idx == 29:
                print("--- [End of First 30 Captions] ---\n")

    # Save the final inference results to a pkl file
    with open(args.out_pkl, "wb") as f:
        pickle.dump(results, f)

    print("Done.")


if __name__ == "__main__":
    main()