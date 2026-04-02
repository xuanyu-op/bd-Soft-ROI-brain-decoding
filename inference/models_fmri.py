import os
import sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perceiver import PerceiverResampler
from neuro_informed_attn_test import NeuroscienceInformedAttention


class Perceiver(nn.Module):
    # Used to further compress the token features produced by NIA and project them into the target hidden space required by the LLM
    def __init__(self, patch_embed_dim=1024, hidden_size=4096, num_latents=256):
        super().__init__()
        self.ln_vision = nn.LayerNorm(patch_embed_dim)
        self.llm_proj = nn.Linear(patch_embed_dim, hidden_size)
        self.perceiver = PerceiverResampler(
            dim=patch_embed_dim,
            dim_head=96,
            depth=6,
            heads=16,
            num_latents=num_latents,
            num_media_embeds=1
        )

    def forward(self, image_features):
        # Apply LayerNorm first, then aggregate features with the PerceiverResampler, and finally project them to the target hidden dimension
        image_features = self.ln_vision(image_features)
        inputs_llm = self.perceiver(image_features)
        return self.llm_proj(inputs_llm)


class BrainROI(nn.Module):
    # BrainROI encoder used during inference, consisting of the NIA module and the Perceiver projection module
    def __init__(
        self,
        hidden_dim,
        out_dim,
        num_latents,
        softroi_root,
        roi_root,
        atlas_names,
        atlas_labels,
        **nia_kwargs
    ):
        super().__init__()

        # Build the neuroscience-informed attention module to encode raw fMRI signals into token representations
        self.nia = NeuroscienceInformedAttention(
            softroi_root=softroi_root,
            atlas_names=atlas_names,
            atlas_labels=atlas_labels,
            n_fmri_tokens=num_latents,
            token_dim=hidden_dim,
            hidden_dim=hidden_dim,
            roi_root=roi_root,
            **nia_kwargs
        )

        # Further compress the NIA output and map it to the target dimension
        self.perceiver = Perceiver(
            patch_embed_dim=hidden_dim,
            hidden_size=out_dim,
            num_latents=num_latents
        )

    def forward(self, x, subject):
        # First extract tokens through NIA, then pass them to the Perceiver to obtain the final output representation
        tokens_from_nia = self.nia(x, subject=subject)
        return self.perceiver(tokens_from_nia)


def build_atlas_labels(softroi_root, atlas_names):
    # Read the global label file for each atlas to determine the label dimension required for model initialization
    print("  - Preparing Atlas Configurations for the fMRI Encoder...")
    try:
        atlas_labels = {
            name: int(
                np.load(
                    os.path.join(softroi_root, f"{name}_label_ids_global.npy")
                ).shape[0]
            )
            for name in atlas_names
        }
        print(f"  - Loaded global label counts for atlases: {atlas_labels}")
        return atlas_labels
    except Exception as e:
        print(f"[ERROR] Could not load global label files from '{softroi_root}'. Error: {e}")
        exit(1)


def build_brainroi_from_args(args, atlas_labels, device):
    # Collect the NIA-related configuration from command-line arguments
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

    # Build the BrainROI model
    model = BrainROI(
        hidden_dim=1024,
        out_dim=args.feat_dim,
        num_latents=args.num_latents,
        softroi_root=args.softroi_root,
        roi_root=args.roi_root,
        atlas_names=args.atlas_names,
        atlas_labels=atlas_labels,
        **nia_kwargs
    )

    # Load the trained model weights
    print(f"Loading BrainROI weights from {args.brainroi_path}...")
    checkpoint = torch.load(args.brainroi_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Return the model moved to the target device and switched to evaluation mode
    return model.to(device).eval()