# Unified-Multimodal-Brain-Decoding-via-Cross-Subject-Soft-ROI-Fusion
- `neuro_informed_attn_test.py`: the core neuroscience-informed fMRI encoder, which combines voxel coordinate encoding, multi-atlas soft-ROI priors, and an attention-based aggregation mechanism to transform variable-length fMRI voxel signals into fixed-length fMRI token representations.
- `perceiver.py`: Perceiver-based resampling module for compressing variable-length input features into fixed latent tokens.
- `utils.py`: General utility module. It provides common functions such as random seed setup, NaN loss checking, parameter counting, NSD WebDataset downloading, and dataloader construction.


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
