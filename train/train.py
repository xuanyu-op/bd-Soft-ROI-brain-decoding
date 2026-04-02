import os
import sys
import json
import numpy as np
from tqdm import tqdm
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from accelerate import Accelerator
import warnings

import configs
import data_nsd
import models_nia
import optim_and_loss
import trainer



def main():
    # Configure the global runtime environment
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize Accelerator for distributed/mixed-precision training
    accelerator = Accelerator(split_batches=False, mixed_precision='fp16')
    device = accelerator.device
    num_devices = torch.cuda.device_count() or 1
    num_workers = max(0, int(os.getenv("NUM_WORKERS", str(num_devices))))

    accelerator.print(f"PID: {os.getpid()}, Device: {device}, Workers: {num_workers}")
    accelerator.print(f"State: {accelerator.state}")

    # Parse command-line arguments and create the output directory
    args = configs.parse_args()
    outdir = configs.setup_run(args, accelerator)

    # Enable image augmentation if specified in the configuration
    img_augmenter = trainer.get_img_augment() if args.use_image_aug else None

    # Build training and validation dataloaders
    train_dls, val_dls, num_train_samples, num_val_samples = data_nsd.build_dataloaders(
        args, accelerator, num_devices, num_workers
    )

    # Compute the number of steps per epoch and the total number of training steps
    batches_per_subject, steps_per_epoch, total_steps = optim_and_loss.compute_steps(
        args.subjects, num_train_samples, args.batch_size, accelerator.state.num_processes, args.num_epochs
    )
    accelerator.print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Load the global label counts for each atlas
    accelerator.print('\n--- Preparing Atlas Configurations ---')
    try:
        atlas_labels = {
            name: int(np.load(os.path.join(args.softroi_root, f"{name}_label_ids_global.npy")).shape[0])
            for name in args.atlas_names
        }
        accelerator.print(f"Loaded global label counts: {atlas_labels}")
    except Exception as e:
        accelerator.print(f"[ERROR] Could not load global labels: {e}")
        exit(1)

    # Create the image encoder and brain encoder models
    accelerator.print('\n--- Creating Models ---')
    image2emb = models_nia.BrainEncoder(args.clip_model_path)
    image2emb.to(device)

    # Collect attention and fusion-related arguments into a dictionary
    nia_kwargs = {
        'coord_norm': args.coord_norm,
        'fusion_mode': args.fusion_mode,
        'gate_voxel_proj_dim': args.gate_voxel_proj_dim,
        'attn_scale': args.attn_scale,
        'attn_norm': args.attn_norm,
        'attn_tau_init': args.attn_tau_init,
        'attn_tau_learnable': args.attn_tau_learnable,
        'attn_dropout': args.attn_dropout,
        'ffn_dropout': args.ffn_dropout,
    }

    # Create the fMRI-to-embedding model
    voxel2emb = models_nia.BrainROI(
        hidden_dim=1024,
        out_dim=args.feat_dim,
        num_latents=args.num_latents,
        softroi_root=args.softroi_root,
        roi_root=args.roi_root,
        atlas_names=args.atlas_names,
        atlas_labels=atlas_labels,
        **nia_kwargs
    )
    voxel2emb.to(device)

    # Print the number of model parameters on the local main process only
    if accelerator.is_local_main_process:
        utils.count_params(voxel2emb)

    # Set whether each model requires gradient updates
    voxel2emb.requires_grad_(True)
    image2emb.requires_grad_(False)

    # Build the optimizer, learning-rate scheduler, and loss function
    optimizer, max_lr = optim_and_loss.build_optimizer(voxel2emb, args, accelerator)
    lr_scheduler = optim_and_loss.build_scheduler(optimizer, args, total_steps, max_lr)
    loss_fn = optim_and_loss.get_loss_fn(args.recon_loss)

    # Initialize variables for checkpoint resume logic
    epoch_start = 0
    val_losses_hist = []
    best_val_loss = 1e9

    # Resume training from a checkpoint if specified
    if args.resume:
        accelerator.print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        voxel2emb.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint.get('epoch', -1) + 1

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if lr_scheduler is not None and 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except:
                accelerator.print("Warning: Could not load LR scheduler state.")

        val_losses_hist = checkpoint.get('val_losses', [])
        best_val_loss = min(val_losses_hist) if val_losses_hist else 1e9
        accelerator.print(f"Resuming from epoch {epoch_start}")
    else:
        accelerator.print("Starting from scratch")

    # Wrap trainable objects with Accelerator
    voxel2emb, optimizer, lr_scheduler = accelerator.prepare(voxel2emb, optimizer, lr_scheduler)

    # Wrap each subject-specific training and validation dataloader
    prepared_train_dls = {subj: accelerator.prepare(dl) for subj, dl in train_dls.items()}
    prepared_val_dls = {subj: accelerator.prepare(dl) for subj, dl in val_dls.items()}

    # Initialize the checkpoint manager and progress bar
    ckpt_manager = trainer.CheckpointManager(accelerator, voxel2emb, optimizer, lr_scheduler, outdir)
    progress_bar = tqdm(range(epoch_start, args.num_epochs), ncols=120, disable=(not accelerator.is_main_process))

    # Main training loop
    for epoch in progress_bar:
        # Gradually increase dropout during the warmup phase
        trainer.maybe_apply_dropout_warmup(voxel2emb, epoch, args, accelerator)

        # Run one epoch of training
        train_loss = trainer.train_one_epoch(
            epoch, args, accelerator, voxel2emb, image2emb, optimizer, lr_scheduler, loss_fn,
            prepared_train_dls, batches_per_subject, img_augmenter
        )

        # Run one epoch of validation and get subject-wise and macro-average losses
        avg_val_losses, macro_avg_val_loss = trainer.validate_and_plot(
            epoch, args, accelerator, voxel2emb, image2emb, loss_fn, prepared_val_dls, outdir
        )

        # Only the main process handles logging and checkpoint saving
        if accelerator.is_main_process:
            val_losses_hist.append(macro_avg_val_loss)

            if args.ckpt_saving:
                # Save the current best model
                if macro_avg_val_loss < best_val_loss:
                    best_val_loss = macro_avg_val_loss
                    ckpt_manager.save('best', epoch, val_losses_hist)
                    accelerator.print(f'New best val loss: {best_val_loss:.4f}')

                # Save the last checkpoint at intervals or at the final epoch
                if args.save_last and (
                    (args.ckpt_interval > 0 and epoch % args.ckpt_interval == 0) or
                    (epoch == args.num_epochs - 1)
                ):
                    ckpt_manager.save('last', epoch, val_losses_hist)

            # Record the current learning rate and training/validation losses
            current_lr = optimizer.param_groups[0]['lr']
            logs = {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss_macro_avg": macro_avg_val_loss,
                "train/lr": current_lr
            }
            logs.update(avg_val_losses)
            progress_bar.set_postfix(**logs)

            # Append logs to the log file
            with open(os.path.join(outdir, 'logs.txt'), 'a') as f:
                f.write(json.dumps(logs) + '\n')

        # Clean up memory after each epoch
        trainer.memory_cleanup(accelerator)

    # Save the best checkpoint again at the end if needed
    if args.save_at_end and args.ckpt_saving and accelerator.is_main_process:
        ckpt_manager.save('best', args.num_epochs, val_losses_hist)

    # Save final plots such as the validation loss curve
    if accelerator.is_main_process:
        trainer.finalize_plots(val_losses_hist, outdir)


if __name__ == "__main__":
    main()
