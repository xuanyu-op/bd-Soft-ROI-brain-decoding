import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops_exts import rearrange_many


def exists(val):
    # Check whether the variable is not None
    return val is not None


def FeedForward(dim, mult=4):
    # Standard feed-forward network:
    # LayerNorm -> expansion linear layer -> GELU -> projection linear layer
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    """
    Cross-Attention module.

    Function:
    - Uses latent tokens as Query
    - Uses the input features together with the latent tokens to construct Key/Value
    - Updates the latent representations through multi-head attention
    """

    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        # Apply normalization separately to the input features and the latent tokens
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        # Query is generated from latent tokens
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        # Key/Value are generated from the concatenation of input features and latent tokens
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        # Project the multi-head output back to the original dimension
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
        - x: input features, usually with shape [B, M, N, D]
        - latents: latent tokens, usually with shape [B, M, L, D]

        Returns:
        - Updated latent representations
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        # Generate Query from latent tokens
        q = self.to_q(latents)

        # Concatenate input features and latent tokens along the token dimension
        # and use them together to generate Key / Value
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Rearrange tensors into the format required for multi-head attention
        q, k, v = rearrange_many(
            (q, k, v),
            'b t n (h d) -> b h t n d',
            h=h
        )

        # Scale Query for numerical stability
        q = q * self.scale

        # Compute attention scores
        sim = einsum('... i d, ... j d -> ... i j', q, k)

        # Stabilize softmax computation
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # Weighted sum of Value vectors
        out = einsum('... i j, ... j d -> ... i d', attn, v)

        # Merge attention heads back together
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)

        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver resampling module.

    Function:
    - Accepts an input token sequence
    - Adds media positional embeddings to the input
    - Uses a set of learnable latent tokens to attend to the input through multiple Cross-Attention layers
    - Outputs a fixed number of latent tokens
    """

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_media_embeds=4,
        ff_mult=4
    ):
        super().__init__()

        # Learnable latent tokens used as the target compressed representation
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Learnable positional embeddings for different media segments
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        # Stack multiple Attention + FeedForward layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        # Final normalization applied to the output
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
        - x:
            If single-segment input, shape can be [B, N, D]
            If multi-segment input, shape can be [B, M, N, D]

        Returns:
        - If the input is single-segment, output shape is [B, L, D]
        - If the input is multi-segment, output shape is [B, M, L, D]
        """
        # If the input is 3D, expand it to the single-media-segment format
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        # Add the corresponding positional embedding to each media segment
        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        # Expand learnable latent tokens across batch and media dimensions
        latents = repeat(
            self.latents,
            'n d -> b m n d',
            b=x.shape[0],
            m=x.shape[1]
        )

        # Iteratively update latent tokens through multiple layers
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # Normalize the final output
        res = self.norm(latents)

        # If there is only one media segment, remove that dimension
        if res.ndim == 4:
            res = res.squeeze(1)

        return res