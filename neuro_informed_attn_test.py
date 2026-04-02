import os
import json
from functools import partial

import nibabel as nib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.utils import safe_get_source
import rff


class RMSNorm(nn.Module):
    """
    RMSNorm normalization layer for optional attention normalization.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class MaskRegistered(nn.Module):
    """
    Load and register the visual masks and affine matrices for each subject. By default, subjects 1, 2, 5, and 7 are supported.
    """
    def __init__(self, roi_root: str):
        super().__init__()
        for subj in [1, 2, 5, 7]:
            path = os.path.join(roi_root, f"subj0{subj}/nsdgeneral.nii.gz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Visual mask not found at {path}.")

            img = nib.load(path)
            arr = img.get_fdata()

            m = (arr == 1)
            visual_mask = torch.from_numpy(m)
            affine = torch.from_numpy(img.affine.astype(np.float32))

            self.register_buffer(f"visual_mask_subject_{subj}", visual_mask)
            self.register_buffer(f"affine_subject_{subj}", affine)

    def get_mask(self, subject):
        """
        Retrieve the corresponding visual mask based on the subject ID or subject string.
        """
        if isinstance(subject, list):
            subject = safe_get_source(subject)
        if not isinstance(subject, str) or not subject.startswith('subject_'):
            subject = f'subject_{subject}'
        return getattr(self, f"visual_mask_{subject}")

    def get_affine(self, subject):
        """
        Retrieve the corresponding affine matrix based on the subject ID or subject string.
        """
        if isinstance(subject, list):
            subject = safe_get_source(subject)
        if not isinstance(subject, str) or not subject.startswith('subject_'):
            subject = f'subject_{subject}'
        return getattr(self, f"affine_{subject}")

    def get_masks(self, subjects):
        """
        Retrieve the visual masks for multiple subjects in batch.
        """
        processed_subjects = []
        for subject in subjects:
            if not isinstance(subject, str) or not subject.startswith('subject_'):
                processed_subjects.append(f'subject_{subject}')
            else:
                processed_subjects.append(subject)
        return [getattr(self, f"visual_mask_{s}") for s in processed_subjects]


class NeuroscienceInformedAttentionLayer(nn.Module):
    """
    Neuroscience-informed attention layer.

    Inputs:
        values: (B, L), scalar value for each voxel
        keys: (B, L, H), key features corresponding to each voxel
    Internal: learnable queries: (1, S, H)
    Output: (B, S)
    """

    def __init__(
        self,
        size: int,
        rank: int = 512,
        scale: str = 'sqrt',
        norm: str = 'layernorm',
        tau_init: float = 1.0,
        tau_learnable: bool = False
    ):
        super().__init__()
        self.size = size
        self.rank = rank
        self.scale = scale == 'sqrt'

        self.query_embeddings = nn.Parameter(torch.empty(1, size, rank))
        nn.init.xavier_uniform_(self.query_embeddings)

        if norm == 'layernorm':
            self.q_norm = nn.LayerNorm(rank)
            self.k_norm = nn.LayerNorm(rank)
        elif norm == 'rmsnorm':
            self.q_norm = RMSNorm(rank)
            self.k_norm = RMSNorm(rank)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        if tau_learnable:
            self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        else:
            self.register_buffer('tau', torch.tensor(float(tau_init)))

    def forward(self, values: torch.Tensor, keys: torch.Tensor, context=None):
        """
        Use learnable queries to compute attention scores over the keys, and then perform weighted aggregation over the voxel scalar values.
        """
        assert values.dim() == 2, f"values must have shape (B, L), but got {tuple(values.shape)}"
        assert keys.dim() == 3, f"keys must have shape (B, L, H), but got {tuple(keys.shape)}"

        B, L = values.shape
        assert keys.size(0) == B and keys.size(1) == L, \
            f"The first two dimensions of keys must match values, got keys={tuple(keys.shape)} vs values={tuple(values.shape)}"

        query = self.q_norm(self.query_embeddings)
        keys = self.k_norm(keys)

        query = query.unsqueeze(1).expand(-1, B, -1, -1)
        attn_logits = (query @ keys.transpose(1, 2).unsqueeze(0))[0]

        if self.scale:
            attn_logits = attn_logits / (self.rank ** 0.5)

        attn_logits = attn_logits / torch.clamp(self.tau, min=1e-2)
        weight = attn_logits.softmax(dim=-1)

        out = (values.unsqueeze(1) @ weight.transpose(1, 2)).squeeze(1)
        return out


class NeuroscienceInformedAttention(MaskRegistered):
    """
    Main module:
    1. Load soft-ROI information.
    2. Apply positional encoding to voxel coordinates.
    3. Fuse ROI representations from multiple atlases.
    4. Perform aggregation through the neuroscience-informed attention layer.
    5. Output fixed-length fMRI token representations.
    """

    def __init__(
        self,
        softroi_root: str,
        atlas_names: list,
        atlas_labels: dict,
        roi_embed_dim: int = 32,
        atlas_dropout_p: float = 0.2,
        n_fmri_tokens: int = 256,
        token_dim: int = 1024,
        rank: int = 128,
        pe_method: str = "gauss",
        hidden_dim: int = 1024,
        n_mlp_layers: int = 4,
        norm_type: str = "ln",
        act_first: bool = False,
        roi_root: str = './roi',
        coord_norm: str = 'unit',
        fusion_mode: str = 'concat',
        gate_voxel_proj_dim: int = 64,
        attn_scale: str = 'sqrt',
        attn_norm: str = 'layernorm',
        attn_tau_init: float = 1.0,
        attn_tau_learnable: bool = False,
        attn_dropout: float = 0.5,
        ffn_dropout: float = 0.15,
    ):
        super().__init__(roi_root=roi_root)

        self.n_fmri_tokens = n_fmri_tokens
        self.pe_method = pe_method
        self.fusion_mode = fusion_mode
        self.atlas_names = atlas_names
        self.softroi_root = softroi_root
        self.coord_norm = coord_norm

        self.soft_roi_cache = {}
        self.coord_cache = {}

        out_dim = n_fmri_tokens * token_dim

        norm_func = (
            partial(nn.BatchNorm1d, num_features=hidden_dim)
            if norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=hidden_dim)
        )
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == "bn" else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)

        self.neuro_informed_attn = NeuroscienceInformedAttentionLayer(
            size=hidden_dim,
            rank=rank,
            scale=attn_scale,
            norm=attn_norm,
            tau_init=attn_tau_init,
            tau_learnable=attn_tau_learnable,
        )

        if self.pe_method == "gauss":
            self.coords_encoding = rff.layers.GaussianEncoding(
                sigma=5.0,
                input_size=3,
                encoded_size=192
            )
            self.coords_transform = nn.Sequential(
                nn.Linear(192 * 2, rank),
                nn.GELU(),
                nn.Linear(rank, rank),
                nn.GELU(),
                nn.Linear(rank, rank)
            )
        else:
            raise NotImplementedError(f"{pe_method} PE not implemented.")

        self.roi_mappings = nn.ModuleDict({
            name: nn.Linear(num_labels, roi_embed_dim, bias=False)
            for name, num_labels in atlas_labels.items()
        })
        for param in self.roi_mappings.values():
            nn.init.xavier_uniform_(param.weight)

        fused_roi_dim = roi_embed_dim * len(atlas_names)

        if self.fusion_mode == 'gate':
            self.gate_mlp = nn.Sequential(
                nn.Linear(roi_embed_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.atlas_dropout = nn.Dropout(atlas_dropout_p)
            self.region_feature_project = nn.Linear(rank + roi_embed_dim, rank)

        elif self.fusion_mode == 'concat':
            self.region_feature_project = nn.Linear(rank + fused_roi_dim, rank)

        elif self.fusion_mode == 'gate_voxel':
            self.gate_voxel_proj = nn.Linear(roi_embed_dim, gate_voxel_proj_dim)
            self.gate_voxel_mlp = nn.Sequential(
                nn.Linear(gate_voxel_proj_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
            self.region_feature_project = nn.Linear(rank + roi_embed_dim, rank)

        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.neuro_informed_attn_post = nn.Sequential(
            *[item() for item in act_and_norm],
            self.attn_dropout,
        )

        self.n_mlp_layers = n_mlp_layers
        self.mlp = nn.ModuleList()
        for _ in range(self.n_mlp_layers):
            ffn_dropout_layer = nn.Dropout(ffn_dropout)
            self.mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                *[item() for item in act_and_norm],
                ffn_dropout_layer,
            ))

        self.head = nn.Linear(hidden_dim, out_dim)

    def _resolve_softroi_subdir(self, subject_str: str) -> str:
        """
        Resolve the corresponding softROI subdirectory name based on the input subject identifier.
        Compatible with multiple directory naming conventions.
        """
        try:
            if isinstance(subject_str, (list, tuple)):
                subject_str = subject_str[0]
            s = str(subject_str)
            sid = int(s.split('_')[-1]) if 'subject_' in s else int(s)
        except Exception:
            return str(subject_str)

        candidates = [f"subject_{sid}", f"subj{sid:02d}", f"subj0{sid}"]
        for cand in candidates:
            p = os.path.join(self.softroi_root, cand)
            if os.path.isdir(p):
                return cand
        return f"subject_{sid}"

    def _get_normalized_coords(self, subject_str, mask):
        """
        Get voxel coordinates within the mask and normalize them according to the specified method.
        The results are cached to avoid repeated computation.
        """
        device = mask.device
        cache_key = f"{subject_str}_{self.coord_norm}"
        if cache_key in self.coord_cache:
            return self.coord_cache[cache_key].to(device)

        ix, iy, iz = torch.where(mask)
        coords_idx = torch.stack([ix, iy, iz], dim=1).float()

        if self.coord_norm == 'unit':
            sx, sy, sz = mask.shape
            coords_idx[:, 0] = 2 * (coords_idx[:, 0] / max(1, sx - 1)) - 1
            coords_idx[:, 1] = 2 * (coords_idx[:, 1] / max(1, sy - 1)) - 1
            coords_idx[:, 2] = 2 * (coords_idx[:, 2] / max(1, sz - 1)) - 1

        elif self.coord_norm == 'mm':
            print("Warning: 'mm' coord_norm not fully implemented. Falling back to 'none'.")
            pass

        elif self.coord_norm == 'none':
            pass

        else:
            raise ValueError(f"Unknown coord_norm: {self.coord_norm}")

        self.coord_cache[cache_key] = coords_idx.cpu()
        return coords_idx.to(device)

    def forward(self, voxels, **kwargs):
        """
        Forward pass:
        1. Retrieve the mask based on the subject.
        2. If the input is a 4D grid, first flatten it into a sequence of voxels within the mask.
        3. Compute coordinate encodings.
        4. Load and map the soft-ROI of each atlas.
        5. Perform multi-atlas fusion.
        6. Pass through the attention layer, MLP, and output head to obtain fixed-length tokens.
        """
        B = voxels.size(0)
        subject_str = safe_get_source(kwargs["subject"])
        mask = self.get_mask(kwargs["subject"])

        if voxels.dim() > 2:
            voxels = voxels.masked_select(mask.unsqueeze(0)).view(B, -1)
        N_target = voxels.size(1)

        coords_idx = self._get_normalized_coords(subject_str, mask)
        coords_batched = coords_idx.unsqueeze(0).expand(B, -1, -1)
        encoding = self.coords_encoding(coords_batched)
        coords = self.coords_transform(encoding)

        E_roi_per_atlas = []
        for name in self.atlas_names:
            subdir = self._resolve_softroi_subdir(subject_str)

            cache_key = f"{subdir}_{name}"
            if cache_key in self.soft_roi_cache:
                R_cpu = self.soft_roi_cache[cache_key]
            else:
                r_pt = os.path.join(self.softroi_root, subdir, f"{name}_R.pt")
                r_npz = os.path.join(self.softroi_root, subdir, f"{name}_R.npz")

                if os.path.exists(r_pt):
                    R_cpu = torch.load(r_pt, map_location='cpu').float()
                elif os.path.exists(r_npz):
                    R_cpu = torch.from_numpy(np.load(r_npz)["R"]).float()
                else:
                    raise FileNotFoundError(
                        f"Soft ROI not found: {r_pt} or {r_npz}. "
                        f"Check --softroi_root and subject mapping (expected subdir='{subdir}')."
                    )

                self.soft_roi_cache[cache_key] = R_cpu

            R_a = R_cpu.to(voxels.device)

            K_expected = self.roi_mappings[name].in_features
            if R_a.shape[1] != K_expected:
                raise RuntimeError(
                    f"[{name}] column mismatch: R cols={R_a.shape[1]} vs Linear.in_features={K_expected} "
                    f"(check whether <atlas>_label_ids_global.npy matches)."
                )

            rows = R_a.shape[0]
            if rows == N_target:
                R_use = R_a
            elif rows > N_target:
                R_use = R_a[:N_target, :]
            else:
                pad = torch.zeros(
                    N_target - rows,
                    R_a.shape[1],
                    device=R_a.device,
                    dtype=R_a.dtype
                )
                R_use = torch.cat([R_a, pad], dim=0)

            E_roi = self.roi_mappings[name](R_use)
            E_roi = E_roi.unsqueeze(0).expand(B, -1, -1)
            E_roi_per_atlas.append(E_roi)

        if self.fusion_mode == 'concat':
            E_fused = torch.cat(E_roi_per_atlas, dim=-1)
            keys = self.region_feature_project(torch.cat([coords, E_fused], dim=-1))

        elif self.fusion_mode == 'gate':
            pooled_E = torch.stack([E.mean(1) for E in E_roi_per_atlas], dim=1)
            scores = self.gate_mlp(pooled_E).squeeze(-1)

            if self.training:
                scores = self.atlas_dropout(scores)

            gate_weights = nn.functional.softmax(scores, dim=-1)
            stacked_E = torch.stack(E_roi_per_atlas, dim=1)
            g = gate_weights.unsqueeze(-1).unsqueeze(-1)
            E_fused = (g * stacked_E).sum(dim=1)
            keys = self.region_feature_project(torch.cat([coords, E_fused], dim=-1))

        elif self.fusion_mode == 'gate_voxel':
            stacked_E = torch.stack(E_roi_per_atlas, dim=2)
            E_proj = self.gate_voxel_proj(stacked_E)
            gate_logits = self.gate_voxel_mlp(E_proj).squeeze(-1)
            gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1)
            E_fused = (stacked_E * gate_weights).sum(dim=2)
            keys = self.region_feature_project(torch.cat([coords, E_fused], dim=-1))

        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        values = voxels
        assert keys.size(0) == values.size(0) and keys.size(1) == values.size(1), \
            f"The B and L dimensions of keys (B, L, *) must match those of values (B, L), got {tuple(keys.shape)} vs {tuple(values.shape)}"

        x = self.neuro_informed_attn(values, keys)
        x = self.neuro_informed_attn_post(x)

        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x

        x = x.reshape(len(x), -1)
        x = self.head(x)
        x = x.reshape(values.size(0), self.n_fmri_tokens, -1)
        return x