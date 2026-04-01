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



