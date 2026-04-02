# Unified-Multimodal-Brain-Decoding-via-Cross-Subject-Soft-ROI-Fusion

Please first install all dependencies listed in requirements.txt. Then download the required ROI, fMRI, and other related data from the NSD dataset. Before running training or inference, make sure that all data paths, model paths, and atlas-related directories are correctly configured. The required data can be obtained from the following source: https://huggingface.co/datasets/BoltzmachineQ/brain-instruction-tuning/tree/main.

The softroi and roi files can be downloaded directly from this GitHub repository. Alternatively, you can generate the softroi files yourself by running build_softroi_all_subjects.py.

## Training
Stage 1 
Run the following command for the first-stage training:

```bash
python train.py \
  --model_name  \
  --data_path  \
  --clip_model_path  \
  --model_save_path  \
  --subjects  \
  --softroi_root  \
  --roi_root  \
  --atlas_names  \
  --feat_dim  \
  --batch_size  \
  --num_epochs  \
  --max_lr  \
  --recon_loss  \
  --fusion_mode  \
  --coord_norm  \
  --attn_scale  \
  --attn_norm  \
  --attn_tau_init  \
  --no-attn_tau_learnable \
  --attn_dropout  \
  --ffn_dropout  \
  --dropout_warm_epochs  \
  --no-plot_umap \
  --ckpt_saving \
  --no-save_last \
  --no-save_at_end
  ```
Stage 2 Fine-tuning(Strongly recommend)

After completing the first-stage training, further fine-tune the best checkpoint from the previous stage using MSE loss:

```bash
python train.py \
  --model_name  \
  --data_path  \
  --clip_model_path  \
  --model_save_path  \
  --subjects  \
  --softroi_root  \
  --roi_root  \
  --atlas_names  \
  --feat_dim  \
  --batch_size  \
  --num_epochs  \
  --max_lr  \
  --fusion_mode  \
  --coord_norm  \
  --attn_scale  \
  --attn_norm  \
  --attn_tau_init  \
  --no-attn_tau_learnable \
  --attn_dropout  \
  --ffn_dropout  \
  --dropout_warm_epochs  \
  --recon_loss mse \
  --resume (the best pth file from last training round) \
  --no-plot_umap \
  --ckpt_saving \
  --no-save_last \
  --no-save_at_end
  ```
## Inference

After training, run the following command for inference:
```bash
python inference.py \
  --shikra_path  \
  --brainroi_path  \
  --clip_model_path  \
  --adapter_path  \
  --feat_dim  \
  --data_path  \
  --out_pkl  \
  --prompt "" \
  --subj  \
  --seed  \
  --prompt_opt_iters  \
  --optimizer_model_path  \
  --stageA_precision  \
  --rank_metric  \
  --coco_captions_path  \
  --prompt_pool_size  \
  --num_new_prompts_per_iter  \
  --optimizer_temperature  \
  --optimizer_top_p  \
  --optimizer_max_new_tokens \
  --prompt_out  \
  --eval_image_dir  \
  --softroi_root  \
  --roi_root  \
  --atlas_names  \
  --fusion_mode  \
  --num_latents  \
  --coord_norm  \
  --attn_scale  \
  --attn_norm  \
  --attn_tau_init  \
  --no-attn_tau_learnable \
  --attn_dropout  \
  --ffn_dropout  \
  --num_beams  \
  --num_return_sequences  \
  --max_new_tokens  \
  --no_repeat_ngram_size  \
  --length_penalty 
```


# Coding files


- `neuro_informed_attn_test.py`: the core neuroscience-informed fMRI encoder, which combines voxel coordinate encoding, multi-atlas soft-ROI priors, and an attention-based aggregation mechanism to transform variable-length fMRI voxel signals into fixed-length fMRI token representations.
- `perceiver.py`: Perceiver-based resampling module for compressing variable-length input features into fixed latent tokens.
- `utils.py`: General utility module. It provides common functions such as random seed setup, NaN loss checking, parameter counting, NSD WebDataset downloading, and dataloader construction.

## Auxiliary Code Overview

- `build_softroi_all_subjects.py`: Builds aligned Soft-ROI membership matrices for all subjects from atlas NIfTI files.
This script automatically detects subject folders, finds the atlases shared across subjects, resamples atlas label maps to each subject’s EPI grid, performs global ROI label alignment, and constructs subject-specific ROI membership matrices R. It supports both hard one-hot assignment and distance-based soft assignment (soft-edt). The outputs include globally aligned ROI label indices, per-subject Soft-ROI matrices, voxel indices, and metadata files, which are used as preprocessing inputs for cross-subject brain decoding.

- `check_voxel_order.py`: Checks whether voxel_indices.npy is fully consistent with the voxel order defined by the subject mask in nsdgeneral.nii.gz.
This script verifies three aspects: voxel count consistency, voxel set consistency, and voxel order consistency. It can also report mismatched samples and build a permutation mapping between the saved voxel order and the flattened mask order. This utility is mainly used to ensure that voxel indices, ROI matrices, and fMRI features are perfectly aligned before model training or inference.

- `evaluate_captions.py`: Evaluates generated captions using both standard language-generation metrics and CLIP-based semantic similarity metrics.
Given a .pkl file of generated captions, a reference caption JSON file, and the corresponding image directory, this script computes BLEU-1/2/3/4, ROUGE-L, CIDEr, CLIP Score, and Reference-based CLIP Score. It provides a standalone evaluation pipeline for quantitative comparison of captioning results in brain decoding experiments.

- `view_diffrence.py`: Exports aligned reference–candidate caption pairs into a plain text file for manual inspection.
This script reads generated captions from a .pkl file and reference captions from a JSON file, matches them by image ID, and writes the results to a tab-separated .txt file in the format:
image_id<TAB>reference_caption<TAB>generated_caption

## Training Code Overview

- `train.py`: Entry-point script for training. It initializes the Accelerator, parses command-line arguments, builds the dataloaders, loads the global label counts for each atlas, creates the BrainEncoder and BrainROI models, configures the optimizer and scheduler, and runs the full epoch-level training and validation pipeline.
- `trainer.py`: Training process management module. It includes checkpoint saving, image augmentation, dropout warmup, single-epoch training, validation, UMAP visualization, memory cleanup, and final loss curve saving.
- `models_nia.py`: Model definition file. BrainEncoder uses a frozen CLIP Vision model to extract image patch tokens, while BrainROI combines NeuroscienceInformedAttention and PerceiverResampler to map fMRI signals into the target feature space.
- `data_nsd.py`: Multi-subject NSD data loading module. It builds the training and validation dataloaders for each subject and organizes the corresponding sample count information. In the current implementation, the training set consists of the train shards plus the val shard, while validation uses the test shards.
- `optim_and_loss.py`: Module for optimizer setup, loss functions, and training step calculation. It supports AdamW, OneCycleLR/LinearLR, as well as MSE, L1, Huber, Quantile, and Charbonnier losses.
- `configs.py`: Command-line argument and experiment configuration module. It defines training-related paths, model architecture, fusion mode, attention settings, dropout warmup, and checkpoint strategies, and is also responsible for creating the output directory and saving the configuration file.

## Inference Code Overview

- `inference.py`: Main entry script for inference. It parses command-line arguments, loads the test dataset, restores the trained BrainROI encoder, optionally runs prompt optimization, injects fMRI features into the language model, performs constrained caption generation, and saves the final predictions to a `.pkl` file.
- `models_fmri.py`: Inference-side definition of the BrainROI encoder. It wraps the Neuroscience-Informed Attention module and the Perceiver-based projection module, and also provides helper functions for loading atlas label dimensions and restoring trained BrainROI checkpoints.
- `prompt_optimizer.py`: Optional prompt optimization module. It uses a two-stage resource-loading strategy to reduce memory pressure: one stage loads the optimizer LLM for generating new prompts, and the other stage loads the evaluation stack for scoring candidate prompts through the full fMRI-to-caption pipeline. It also saves optimization logs, prompt histories, and score curves.
- `data_loader.py`: Data loading utilities for inference. It builds the WebDataset test pipeline and also creates a lightweight cache of voxel signals, image tensors, and COCO IDs for prompt optimization.
- `common_utils.py`: Shared helper functions used during inference, including random seed setup, voxel preprocessing, unified text-generation argument construction, caption cleaning, robust JSON parsing, and Java runtime checking for optional SPICE evaluation.
- `projector.py`: Utility for loading pretrained `mm_projector` weights. It extracts projector-related parameters from a checkpoint, checks key mismatches, and loads them into the current projection layer in non-strict mode.
- `metrics.py`: Evaluation utilities for caption scoring, including CLIP-S, RefCLIP-S, and standard NLG metrics such as BLEU and CIDEr. It also contains a BLEU-4 candidate reranking helper for analysis utilities.
- `patch_evalcap.py`: Silent patch for `pycocoevalcap`. It suppresses verbose tokenizer and BLEU scorer outputs so that evaluation logs remain clean during inference and prompt optimization.
