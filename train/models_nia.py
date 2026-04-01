import torch
import os
import sys
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPVisionModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from perceiver import PerceiverResampler
from neuro_informed_attn_test import NeuroscienceInformedAttention


class Perceiver(nn.Module):
    """
    Normalize, resample, and linearly project input token features
    into a fixed number of tokens with the target hidden dimension.
    """
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
        # Apply layer normalization to the input features
        image_features = self.ln_vision(image_features)

        # Use PerceiverResampler to compress the input features into a fixed number of latent tokens
        inputs_llm = self.perceiver(image_features)

        # Project the features to the target hidden dimension
        return self.llm_proj(inputs_llm)


class BrainEncoder(nn.Module):
    """
    Extract image features using a frozen CLIP Vision model
    and return the patch token representations from the second-to-last hidden layer.
    """
    def __init__(self, clip_model_path):
        super().__init__()

        # Load the pretrained CLIP Vision model from a local path
        self.clip = CLIPVisionModel.from_pretrained(clip_model_path, local_files_only=True)
        self.clip.eval()

        # Define the image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size=224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # Freeze all CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False

    def encode_image(self, x):
        # Preprocess the input image tensor
        x_preprocessed = self.preprocess(x)

        # Move the image tensor to the same device as the CLIP model
        device = next(self.clip.parameters()).device
        x_preprocessed = x_preprocessed.to(device, non_blocking=True)

        # Extract hidden states from the CLIP Vision model
        outputs = self.clip(pixel_values=x_preprocessed, output_hidden_states=True)

        # Return patch tokens from the second-to-last hidden layer, excluding the CLS token
        return outputs.hidden_states[-2][:, 1:]


class BrainROI(nn.Module):
    """
    The fMRI encoding branch.
    First extracts token representations from brain signals using NeuroscienceInformedAttention, then maps them to the target feature space through the Perceiver module.
    """
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

        # Build the neuroscience-informed attention module
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

        # Build the Perceiver-based projection module
        self.perceiver = Perceiver(
            patch_embed_dim=hidden_dim,
            hidden_size=out_dim,
            num_latents=num_latents
        )

    def forward(self, x, subject):
        # Generate token representations from brain signals using the NIA module
        tokens_from_nia = self.nia(x, subject=subject)

        # Map the tokens to the final output space through the Perceiver module
        final_tokens = self.perceiver(tokens_from_nia)

        return final_tokens