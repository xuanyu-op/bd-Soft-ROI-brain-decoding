import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import ctypes
import utils


class CheckpointManager:
    # Save checkpoints during training
    def __init__(self, accelerator, model, optimizer, scheduler, outdir):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.outdir = outdir

    # Save different types of checkpoints according to the tag
    def save(self, tag, epoch, val_losses_hist):
        ckpt_path = os.path.join(self.outdir, f'{tag}.pth')
        self.accelerator.print(f'\nSaving {ckpt_path}', flush=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        try:
            state_dict = {
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
            }

            # The best checkpoint stores extra states for later resume
            if tag == "best":
                if self.scheduler is not None:
                    state_dict['lr_scheduler'] = self.scheduler.state_dict()
                state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                state_dict['val_losses'] = val_losses_hist

            torch.save(state_dict, ckpt_path)
        except Exception as e:
            self.accelerator.print(f"Couldn't save checkpoint due to an error: {e}")


# Build the image augmentation pipeline
def get_img_augment():
    import kornia
    from kornia.augmentation.container import AugmentationSequential

    return AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.6, 1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.5),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3), data_keys=["input"],
    )


# Gradually increase dropout during the early training stage
def maybe_apply_dropout_warmup(model, epoch, args, accelerator):
    if args.dropout_warm_epochs > 0:
        scale = min(1.0, epoch / args.dropout_warm_epochs)
        unwrapped_model = accelerator.unwrap_model(model)

        # Update the attention dropout rate
        if hasattr(unwrapped_model.nia, 'attn_dropout'):
            unwrapped_model.nia.attn_dropout.p = args.attn_dropout * scale

        # Update the dropout rate in each MLP block
        if hasattr(unwrapped_model.nia, 'mlp'):
            for mlp_block in unwrapped_model.nia.mlp:
                if isinstance(mlp_block[-1], nn.Dropout):
                    mlp_block[-1].p = args.ffn_dropout * scale

        if accelerator.is_main_process and epoch < args.dropout_warm_epochs:
            accelerator.print(f"Epoch {epoch}: Dropout warmup scale={scale:.2f}")


# Run one epoch of training
def train_one_epoch(epoch, args, accelerator, voxel2emb, image2emb, optimizer, scheduler, loss_fn,
                    prepared_train_dls, batches_per_subject, img_augmenter):
    voxel2emb.train()
    epoch_losses = []

    # Create one iterator for each subject-specific dataloader
    train_iterators = {subj: iter(dl) for subj, dl in prepared_train_dls.items()}

    # Build and shuffle the subject queue according to the batch count of each subject
    subject_queue = []
    for subj in args.subjects:
        subject_queue.extend([subj] * batches_per_subject[subj])
    random.shuffle(subject_queue)

    for train_i, subj in enumerate(subject_queue):
        try:
            voxel, image = next(train_iterators[subj])
        except StopIteration:
            train_iterators[subj] = iter(prepared_train_dls[subj])
            voxel, image = next(train_iterators[subj])

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            # During training, select one of the three repetitions in a cyclic manner
            voxel = voxel[:, train_i % 3].float()

            subject_batch = [f"subject_{subj}"] * voxel.size(0)
            emb_voxel = voxel2emb(voxel, subject=subject_batch)

            # Apply image augmentation if enabled
            if args.use_image_aug and img_augmenter is not None:
                image = img_augmenter(image)

            # The image encoder is frozen and only used to extract image embeddings
            with torch.no_grad():
                emb_image = accelerator.unwrap_model(image2emb).encode_image(image).to(emb_voxel.dtype)

            # Compute the alignment loss
            loss = loss_fn(emb_voxel, emb_image)

            # Check whether the loss becomes NaN
            utils.check_loss(loss)

            accelerator.backward(loss)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


# Run validation and optionally generate UMAP plots
def validate_and_plot(epoch, args, accelerator, voxel2emb, image2emb, loss_fn, prepared_val_dls, outdir):
    voxel2emb.eval()
    val_losses_per_subject = {subj: [] for subj in args.subjects}

    # Buffers used for UMAP visualization
    all_val_emb_voxels, all_val_emb_images, all_val_subjects = [], [], []
    capture_umap = args.plot_umap and (
        epoch == args.num_epochs - 1 or (args.ckpt_interval > 0 and epoch % args.ckpt_interval == 0)
    )

    with torch.no_grad():
        for subj in args.subjects:
            for val_i, (voxel, image) in enumerate(prepared_val_dls[subj]):
                with torch.cuda.amp.autocast():
                    # During validation, average the three repetitions
                    voxel = torch.mean(voxel, axis=1).float()
                    subject_batch = [f"subject_{subj}"] * voxel.size(0)

                    emb_voxel = voxel2emb(voxel, subject=subject_batch)
                    emb_image = accelerator.unwrap_model(image2emb).encode_image(image).to(emb_voxel.dtype)

                    val_loss = loss_fn(emb_voxel, emb_image)
                    val_losses_per_subject[subj].append(val_loss.item())

                    # Cache embeddings for UMAP if needed
                    if capture_umap:
                        ev = emb_voxel.detach().to('cpu')
                        ei = emb_image.detach().to('cpu')
                        all_val_emb_voxels.append(ev)
                        all_val_emb_images.append(ei)
                        all_val_subjects.extend([subj] * len(ev))

    avg_val_losses = {}
    macro_avg_val_loss = 0.0

    # Only the main process computes summary metrics and draws figures
    if accelerator.is_main_process:
        avg_val_losses = {f"val/loss_s{subj}": np.mean(losses) for subj, losses in val_losses_per_subject.items()}
        macro_avg_val_loss = np.mean(list(avg_val_losses.values()))

        if capture_umap:
            _plot_umap(epoch, args.seed, all_val_emb_voxels, all_val_emb_images, all_val_subjects, outdir, accelerator)

    return avg_val_losses, macro_avg_val_loss


# Draw the UMAP projection of validation embeddings
def _plot_umap(epoch, seed, all_val_emb_voxels, all_val_emb_images, all_val_subjects, outdir, accelerator):
    import umap

    accelerator.print('UMAP plotting...')
    if not all_val_subjects:
        accelerator.print('UMAP skipped: no validation batches captured.')
        return

    emb_image_np = torch.cat(all_val_emb_images).flatten(1).cpu().numpy()
    emb_voxel_np = torch.cat(all_val_emb_voxels).flatten(1).cpu().numpy()
    subj_labels = np.array(all_val_subjects)

    combined = np.concatenate((emb_image_np, emb_voxel_np), axis=0)
    source_labels = np.array(['image'] * len(emb_image_np) + [f'subj_{s}' for s in subj_labels])
    unique_sources = sorted(list(set(source_labels)))
    color_map = plt.get_cmap('viridis', len(unique_sources))

    reducer = umap.UMAP(random_state=seed)
    embedding = reducer.fit_transform(combined)

    fig = plt.figure(figsize=(10, 10))
    for i, source_name in enumerate(unique_sources):
        mask = source_labels == source_name
        plt.scatter(embedding[mask, 0], embedding[mask, 1], color=color_map(i), label=source_name, alpha=0.6)

    plt.title(f"UMAP Projection at Epoch {epoch}")
    plt.legend()
    plt.savefig(os.path.join(outdir, f'umap-val-epoch{epoch:03d}.png'))
    plt.close(fig)

    # Release intermediate variables used for plotting
    del emb_image_np, emb_voxel_np, combined, embedding
    gc.collect()


# Perform one round of generic memory cleanup
def memory_cleanup(accelerator):
    accelerator.wait_for_everyone()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# Draw and save the final validation loss curve
def finalize_plots(val_losses_hist, outdir):
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses_hist, label='Macro-Avg Validation Loss (per epoch)')
    plt.legend()
    plt.title('Macro-Avg Validation Loss')
    plt.grid(True)
    plt.savefig(os.path.join(outdir, 'loss_val.png'))
    plt.close()