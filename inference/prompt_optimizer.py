import os
import json
import time
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM, CLIPModel, CLIPProcessor
from torchvision.transforms import ToPILImage
from contextlib import nullcontext
import matplotlib.pyplot as plt
import warnings

import common_utils
import metrics
import models_fmri
import projector


def log_memory_usage(stage_name=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[{stage_name}] CUDA Memory Usage: Allocated = {allocated:.2f} GB, Reserved = {reserved:.2f} GB")


class PromptOptimizer:
    def __init__(self, args, device, persistent_cache):
        self.args = args
        self.device = device
        self.persistent_cpu_cache = persistent_cache
        self.prompt_pool = {}
        self.history = []
        self.full_log = []

        # Save the input/output records of the optimizer LLM in Stage A
        self.stage_a_conversations = []

        # Resource dictionaries loaded separately for different stages
        self.stage_a_assets = {}
        self.stage_b_assets = {}

        self.autocast_context = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
        # Check whether SPICE and its required environment should be enabled
        self.spice_enabled = False
        need_spice = (self.args.rank_metric in {'spice', 'match'}) or getattr(self.args, 'enable_spice', False)

        if need_spice:
            print("SPICE metric requested. Checking dependencies...")
            if not common_utils.check_java_runtime():
                warnings.warn("Java Runtime Environment (JRE) not found. SPICE will be disabled.")
                if self.args.rank_metric == 'spice':
                    self.args.rank_metric = 'cider'
            else:
                corenlp_home = os.environ.get('CORENLP_HOME', '')
                default_cache = os.path.expanduser('~/.cache/pycocoevalcap/stanford-corenlp-3.6.0')

                def _has_jars(p):
                    # Check whether the specified directory contains the required Stanford CoreNLP jar files
                    return os.path.isdir(p) and os.path.exists(os.path.join(p, 'stanford-corenlp-3.6.0.jar'))

                jar_dir = corenlp_home if _has_jars(corenlp_home) else default_cache
                if _has_jars(jar_dir):
                    os.environ.setdefault('SPICE_HOME', jar_dir)
                    self.spice_enabled = True
                    print(f"SPICE enabled. JARs found at {jar_dir}")
                else:
                    warnings.warn(f"SPICE JARs not found. SPICE disabled.")
                    if self.args.rank_metric == 'spice':
                        self.args.rank_metric = 'cider'

        # Initialize the prompt pool
        self._initialize_prompt_pool()

    def _initialize_prompt_pool(self):
        # Initialize the prompt pool from a file, a semicolon-separated string, or the default prompt
        if self.args.prompt_init:
            if os.path.exists(self.args.prompt_init):
                with open(self.args.prompt_init, 'r', encoding='utf-8') as f:
                    initial_prompts = [line.strip() for line in f if line.strip()]
            else:
                initial_prompts = [p.strip() for p in self.args.prompt_init.split(';')]
        else:
            initial_prompts = [self.args.prompt]

        # Require each initial prompt to contain exactly one <image> placeholder
        for p in initial_prompts:
            if p.count('<image>') != 1:
                raise ValueError(f"Initial prompt '{p}' must contain exactly one '<image>' placeholder.")
            self.prompt_pool[p] = {'iter_added': 0}

    def _load_stage_a_assets(self):
        # Load the optimizer LLM and tokenizer used for generating new prompts
        common_utils.print_header(f"Loading Stage A Assets (Optimizer LLM at {self.args.stageA_precision} precision)")
        log_memory_usage("Before Stage A Load")

        model_kwargs = {"device_map": "auto", "trust_remote_code": True, "local_files_only": True}

        # Determine the precision and quantization mode for the Stage A model based on configuration
        if self.args.stageA_precision == '4bit':
            print("  - Using 4-bit quantization (NF4).")
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
        elif self.args.stageA_precision == '8bit':
            print("  - Using 8-bit quantization.")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.args.stageA_precision == 'fp16':
            print("  - Using FP16 precision.")
            model_kwargs["torch_dtype"] = torch.float16
        elif self.args.stageA_precision == 'bf16':
            print("  - Using BF16 precision.")
            model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported stageA_precision mode: '{self.args.stageA_precision}'. "
                f"Supported modes are: '4bit', '8bit', 'fp16', 'bf16'."
            )

        self.stage_a_assets['model'] = AutoModelForCausalLM.from_pretrained(
            self.args.optimizer_model_path, **model_kwargs
        )
        self.stage_a_assets['tokenizer'] = AutoTokenizer.from_pretrained(
            self.args.optimizer_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        log_memory_usage("After Stage A Load")

    def _unload_stage_a_assets(self):
        # Unload Stage A resources and clear GPU memory
        common_utils.print_header("Unloading Stage A Assets")
        log_memory_usage("Before Stage A Unload")
        self.stage_a_assets.clear()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        log_memory_usage("After Stage A Unload")

    def _load_stage_b_assets(self):
        # Load the full inference stack used for prompt evaluation
        common_utils.print_header("Loading Stage B Assets (Evaluation Stack at FP16)")
        log_memory_usage("Before Stage B Load")

        # Build atlas label dimensions and load the BrainROI encoder
        atlas_labels = models_fmri.build_atlas_labels(self.args.softroi_root, self.args.atlas_names)
        voxel2emb = models_fmri.build_brainroi_from_args(self.args, atlas_labels, self.device)
        self.stage_b_assets['voxel2emb'] = voxel2emb

        # Load the Shikra tokenizer and language model
        self.stage_b_assets['tokenizer'] = LlamaTokenizer.from_pretrained(
            self.args.shikra_path,
            padding_side='left',
            local_files_only=True
        )
        self.stage_b_assets['model'] = LlamaForCausalLM.from_pretrained(
            self.args.shikra_path,
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(self.device).eval()
        self.stage_b_assets['im_start_id'] = self.stage_b_assets['tokenizer'].convert_tokens_to_ids("<im_start>")

        # If the fMRI feature dimension is 1024, an additional mm_projector is required
        if self.args.feat_dim == 1024:
            mm_projector = torch.nn.Linear(1024, 4096)
            projector.load_mm_projector_weights(mm_projector, self.args.adapter_path)
            self.stage_b_assets['mm_projector'] = mm_projector.to(self.device).eval()

        # If the ranking metric depends on CLIP, load the CLIP model and processor as well
        if self.args.rank_metric in ['clip', 'align', 'refclip']:
            self.stage_b_assets['clip_model'] = CLIPModel.from_pretrained(
                self.args.clip_model_path,
                local_files_only=True
            ).to(self.device).eval()
            self.stage_b_assets['clip_processor'] = CLIPProcessor.from_pretrained(
                self.args.clip_model_path,
                local_files_only=True
            )
            self.stage_b_assets['ref_text_embedding_cache'] = {}

        log_memory_usage("After Stage B Load")

    def _unload_stage_b_assets(self):
        # Unload Stage B resources and actively clear GPU memory
        common_utils.print_header("Unloading Stage B Assets")
        log_memory_usage("Before Stage B Unload")
        self.stage_b_assets.clear()
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        log_memory_usage("After Stage B Unload")

    def run(self):
        # Execute the full prompt optimization workflow
        common_utils.print_header("Starting Prompt Optimization")

        for i in range(self.args.prompt_opt_iters):
            iter_num = i + 1
            common_utils.print_header(f"Optimization Iteration {iter_num}/{self.args.prompt_opt_iters}")

            # Find prompts in the current pool that have not been scored yet
            prompts_to_evaluate = [p for p, data in self.prompt_pool.items() if 'score' not in data]

            # If there are unscored prompts, load the evaluation stack first
            if prompts_to_evaluate:
                self._load_stage_b_assets()
                self._evaluate_prompts(prompts_to_evaluate)
                self._unload_stage_b_assets()

            # Sort the prompt pool by score
            sorted_pool = sorted(
                self.prompt_pool.items(),
                key=lambda item: item[1].get('score', -1),
                reverse=True
            )
            self.full_log.append({p: d for p, d in sorted_pool})

            # Record the best score and top-5 average score of this iteration
            best_score = sorted_pool[0][1].get('score', -1)
            top5_avg = np.mean([d.get('score', -1) for p, d in sorted_pool[:5]]) if len(sorted_pool) >= 5 else best_score
            self.history.append({'iter': iter_num, 'best_score': best_score, 'top5_avg': top5_avg})

            print(f"Iteration {iter_num} complete. Best score: {best_score:.4f}")

            # Stop optimization if the early stopping condition is triggered
            if self._check_early_stopping():
                print("Plateau detected. Stopping early.")
                break

            # Continue generating new prompts before the final iteration
            if i < self.args.prompt_opt_iters - 1:
                self._load_stage_a_assets()
                new_prompts = self._generate_new_prompts(sorted_pool, iter_num)
                for p in new_prompts:
                    if p not in self.prompt_pool:
                        self.prompt_pool[p] = {'iter_added': iter_num}
                self._unload_stage_a_assets()

        # Perform a final evaluation on all remaining unscored prompts
        final_prompts_to_evaluate = [p for p, data in self.prompt_pool.items() if 'score' not in data]
        if final_prompts_to_evaluate:
            common_utils.print_header("Final Evaluation of Remaining Prompts")
            self._load_stage_b_assets()
            self._evaluate_prompts(final_prompts_to_evaluate)
            self._unload_stage_b_assets()

        # Save all optimization results and return the best prompt
        final_sorted_pool = sorted(
            self.prompt_pool.items(),
            key=lambda x: x[1].get('score', -1),
            reverse=True
        )
        self._save_artifacts(final_sorted_pool)
        return final_sorted_pool[0][0]

    def _check_early_stopping(self):
        # If the best score has barely improved over the last few iterations, treat it as a plateau
        if len(self.history) < 4:
            return False
        last_3_bests = [h['best_score'] for h in self.history[-3:]]
        if last_3_bests[2] < last_3_bests[0] + 0.001:
            return True
        return False

    def _evaluate_prompts(self, prompts_to_evaluate):
        # Split the cached validation data into chunks and evaluate prompts chunk by chunk
        full_data = self.persistent_cpu_cache['cached_features']
        chunk_size = self.args.eval_chunk_size if self.args.eval_chunk_size > 0 else len(full_data)
        data_chunks = [full_data[i:i + chunk_size] for i in range(0, len(full_data), chunk_size)]

        tokenizer = self.stage_b_assets['tokenizer']
        gen_kwargs = common_utils.get_generation_kwargs(self.args, tokenizer)

        # For each prompt to be evaluated, generate outputs and compute scores chunk by chunk, then average them
        for prompt in tqdm(prompts_to_evaluate, desc="Evaluating Prompts"):
            all_scores = []
            for chunk in data_chunks:
                captions, images, refs = self._generate_batch(prompt, chunk, gen_kwargs)
                score_dict = self._calculate_scores(captions, images, refs)
                all_scores.append(score_dict)

            if all_scores:
                final_score = np.mean([s['score'] for s in all_scores])
                final_breakdown = {}
                for k in all_scores[0]['breakdown'].keys():
                    final_breakdown[k] = np.mean([s['breakdown'].get(k, 0) for s in all_scores])
                self.prompt_pool[prompt].update({'score': final_score, 'breakdown': final_breakdown})
            else:
                self.prompt_pool[prompt].update({'score': 0.0})

            bd_str = ", ".join([
                f"{k}: {v:.4f}"
                for k, v in self.prompt_pool[prompt].get('breakdown', {}).items()
            ])
            print(f"  - Prompt: \"{prompt[:30]}...\" | Score: {self.prompt_pool[prompt]['score']:.4f} ({bd_str})")

    def _calculate_scores(self, captions, images, refs):
        # Compute scores according to the currently selected ranking metric
        metric = self.args.rank_metric

        if metric == 'clip':
            s = metrics.calculate_clip_s(
                images, captions,
                self.stage_b_assets['clip_model'],
                self.stage_b_assets['clip_processor'],
                self.device,
                self.args.clip_image_bs,
                self.args.clip_text_bs
            )
            return {'score': s, 'breakdown': {'clip_score': s}}

        if metric == 'refclip':
            s = metrics.calculate_refclip_s(
                captions, refs,
                self.stage_b_assets['clip_model'],
                self.stage_b_assets['clip_processor'],
                self.device,
                self.stage_b_assets['ref_text_embedding_cache']
            )
            return {'score': s, 'breakdown': {'refclip_score': s}}

        if metric == 'align':
            c_score = metrics.calculate_clip_s(
                images, captions,
                self.stage_b_assets['clip_model'],
                self.stage_b_assets['clip_processor'],
                self.device,
                self.args.clip_image_bs,
                self.args.clip_text_bs
            )
            r_score = metrics.calculate_refclip_s(
                captions, refs,
                self.stage_b_assets['clip_model'],
                self.stage_b_assets['clip_processor'],
                self.device,
                self.stage_b_assets['ref_text_embedding_cache']
            )
            score = 0.6 * c_score + 0.4 * r_score
            return {
                'score': score,
                'breakdown': {
                    'clip_score': c_score,
                    'refclip_score': r_score,
                    'align_score': score
                }
            }

        if metric in ['bleu1', 'bleu4', 'cider', 'spice', 'match']:
            nlg_scores = metrics.calculate_nlg_metrics(captions, refs, self.spice_enabled)

            # The match metric is a weighted combination of multiple text-generation metrics
            if metric == 'match':
                if self.spice_enabled:
                    score = 0.25 * nlg_scores['Bleu_4'] + 0.30 * nlg_scores['CIDEr'] + 0.45 * nlg_scores['SPICE']
                else:
                    score = 0.4 * nlg_scores['Bleu_4'] + 0.6 * nlg_scores['CIDEr']
                breakdown = {**nlg_scores, 'match_score': score}
            else:
                key_map = {
                    'bleu1': 'Bleu_1',
                    'bleu4': 'Bleu_4',
                    'cider': 'CIDEr',
                    'spice': 'SPICE'
                }
                score = nlg_scores[key_map[metric]]
                breakdown = {metric: score}

            return {'score': score, 'breakdown': breakdown}

        raise ValueError(f"Unknown metric: {metric}")

    def _generate_new_prompts(self, sorted_pool, iter_num):
        # Use the optimizer LLM to generate new prompt candidates based on historically high-scoring prompts
        print("Generating new prompts...")
        top_prompts = [item[0] for item in sorted_pool[:30]]

        def format_for_llm(prompts):
            # Format prompts and their historical scores into text for the optimizer model
            return "\n".join([
                f"- \"{p}\" (Score: {self.prompt_pool[p].get('score', -1):.3f})"
                for p in prompts
            ])

        base_instruction = (
            f"You are an expert prompt engineer for an AI that decodes fMRI signals into image descriptions. "
            f"Your task is to generate {self.args.num_new_prompts_per_iter} new, diverse, and high-quality prompts for this task. "
            f"Analyze the provided high-performing prompts to understand what works.\n\n"
            f"--- Rules ---\n"
            f"1. Each prompt MUST contain exactly one `<image>` placeholder.\n"
            f"2. Do NOT modify or remove the `<image>` placeholder.\n"
            f"3. Your entire output MUST be a single, valid JSON list of strings. Do not add any explanations, markdown, or text outside the JSON array.\n\n"
            f"--- Top Performing Prompts (Ranked by Score) ---\n"
            f"Here are up to {len(top_prompts)} of the best-performing prompts from all previous evaluations, ranked from highest to lowest score. "
            f"Analyze their structure, wording, and instructions to inform your new creations.\n\n"
            f"{format_for_llm(top_prompts)}\n\n"
            f"Now, generate {self.args.num_new_prompts_per_iter} new prompts based on your analysis. Output ONLY the JSON list:"
        )

        model = self.stage_a_assets['model']
        tokenizer = self.stage_a_assets['tokenizer']

        response_text = "Failed"

        # Try up to 3 times to generate new prompts in valid JSON format
        for i in range(3):
            try:
                messages = [{"role": "user", "content": base_instruction}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
                input_len = model_inputs.input_ids.shape[1]

                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=self.args.optimizer_max_new_tokens,
                    do_sample=True,
                    temperature=self.args.optimizer_temperature,
                    top_p=self.args.optimizer_top_p
                )
                response_ids = generated_ids[0, input_len:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

                new_prompts = common_utils.robust_json_parse(response_text)
                if new_prompts and isinstance(new_prompts, list):
                    validated = [
                        p for p in new_prompts
                        if isinstance(p, str) and p.count('<image>') == 1
                    ]
                    if validated:
                        print(f"LLM generated {len(validated)} valid new prompts.")
                        self.stage_a_conversations.append({
                            'iteration': iter_num,
                            'prompt_to_optimizer': base_instruction,
                            'raw_response': response_text,
                            'status': 'Success'
                        })
                        return validated

                print(f"Invalid JSON/Prompts. Retry {i + 1}...")

            except Exception as e:
                print(f"Generation error {i + 1}: {e}")
                time.sleep(2)

        # If all attempts fail, record the failure log and return an empty list
        self.stage_a_conversations.append({
            'iteration': iter_num,
            'prompt_to_optimizer': base_instruction,
            'raw_response': response_text,
            'status': 'Failed'
        })
        return []

    def _generate_batch(self, prompt, chunk, gen_kwargs):
        # Execute the full fMRI -> caption generation pipeline on one data chunk
        tokenizer = self.stage_b_assets['tokenizer']
        model = self.stage_b_assets['model']
        voxel2emb = self.stage_b_assets['voxel2emb']
        to_pil = ToPILImage()

        # Build the conversation template containing image-token placeholders
        system = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
        )
        user_image = " <im_start>" + "<im_patch>" * self.args.num_latents + "<im_end> "
        user_prompt = prompt.replace('<image>', user_image)
        input_text = system + user_prompt + " ASSISTANT:"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        inputs_embeds = model.model.embed_tokens(input_ids)

        # Locate the starting position of the image tokens so the fMRI features can be inserted
        im_start_id = self.stage_b_assets['im_start_id']
        pos = torch.where(input_ids[0] == im_start_id)[0]
        image_start_token_pos = pos[0].item()
        num_patches = self.args.num_latents

        best_captions, all_images, all_refs = [], [], []

        with torch.no_grad(), self.autocast_context:
            for sample in chunk:
                # Preprocess voxel inputs and extract fMRI features with BrainROI
                voxels = common_utils.preprocess_voxels(sample['voxels_fp16_numpy'], self.device)
                subject_batch = [f"subject_{self.args.subj}"]
                fmri_feats = voxel2emb(voxels, subject=subject_batch)

                # If needed, project 1024-dimensional features into the 4096-dimensional LLM hidden space
                if 'mm_projector' in self.stage_b_assets and self.args.feat_dim == 1024:
                    fmri_feats = self.stage_b_assets['mm_projector'](fmri_feats)

                # Replace the image-token region in the input embeddings with fMRI features
                final_embeds = torch.cat((
                    inputs_embeds[0, :image_start_token_pos + 1],
                    fmri_feats[0],
                    inputs_embeds[0, image_start_token_pos + num_patches + 1:]
                ), dim=0).unsqueeze(0)

                attention_mask = torch.ones(final_embeds.shape[:-1], dtype=torch.long, device=self.device)
                output_ids = model.generate(
                    inputs_embeds=final_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )


                responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                best_cap = common_utils.clean_caption(responses[0]) if len(responses) > 0 else ""

                refs_list = self.persistent_cpu_cache['coco_id_to_captions'][sample['coco_id']]

                best_captions.append(best_cap)
                all_images.append(to_pil(sample['image_tensor'].cpu() / 255.0).convert("RGB"))
                all_refs.append(refs_list)

        return best_captions, all_images, all_refs

    def _save_artifacts(self, sorted_pool):
        # Save all result files generated during the prompt optimization process
        import datetime

        out_pkl_dir = os.path.dirname(self.args.out_pkl)
        time_str = str(int(time.time()))
        opt_dir = os.path.join(out_pkl_dir, f"prompt_optimization_subj{self.args.subj}_{time_str}")
        os.makedirs(opt_dir, exist_ok=True)

        # Save the prompt pool results table
        records = [
            {
                'prompt': p,
                'score': d['score'],
                'iter_added': d.get('iter_added'),
                **d.get('breakdown', {})
            }
            for p, d in sorted_pool
        ]
        df = pd.DataFrame(records)

        try:
            df.to_excel(os.path.join(opt_dir, "prompt_pool_results.xlsx"), index=False)
        except ImportError:
            warnings.warn("`openpyxl` is not installed. Saving results to CSV instead of Excel.")
            df.to_csv(os.path.join(opt_dir, "prompt_pool_results.csv"), index=False)

        # Save prompt pool snapshots for each iteration
        with open(os.path.join(opt_dir, "prompt_history.jsonl"), 'w', encoding='utf-8') as f:
            for i, item in enumerate(self.full_log):
                f.write(json.dumps({
                    'iteration': i + 1,
                    'pool': {str(k): v for k, v in item.items()}
                }, ensure_ascii=False, indent=2) + "\n")

        # Save the Stage A optimizer conversation logs
        with open(os.path.join(opt_dir, "stage_a_optimizer_conversations.jsonl"), 'w', encoding='utf-8') as f:
            for convo in self.stage_a_conversations:
                f.write(json.dumps(convo, ensure_ascii=False, indent=2) + "\n")

        # If historical scores exist, plot the score progression curve
        if self.history:
            history_df = pd.DataFrame(self.history)
            plt.figure(figsize=(10, 5))
            plt.plot(history_df['iter'], history_df['best_score'], marker='o', label='Best Score')
            plt.plot(history_df['iter'], history_df['top5_avg'], marker='x', linestyle='--', label='Top-5 Avg Score')
            plt.xlabel("Iteration")
            plt.ylabel("Score")
            plt.title("Prompt Score Progression")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(opt_dir, "score_history.png"))
            plt.close()

        print(f"Saved optimization artifacts to directory: {opt_dir}")

        # If prompt_out is specified, additionally save the best prompt string separately
        if self.args.prompt_out:
            try:
                best_prompt = sorted_pool[0][0]
                with open(self.args.prompt_out, 'w', encoding='utf-8') as f:
                    f.write(best_prompt)
                print(f"Saved BEST prompt string to file: {self.args.prompt_out}")
            except Exception as e:
                print(f"Error saving prompt to file {self.args.prompt_out}: {e}")