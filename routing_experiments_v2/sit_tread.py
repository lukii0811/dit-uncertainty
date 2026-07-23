"""SiT/REPA-compatible backbone with inference-time TREAD token routing.

The parameter names intentionally follow the official REPA SiT implementation
so its checkpoints can be loaded without renaming the generative backbone.
Routing has no learnable parameters: selected tokens traverse blocks from
``start_layer_idx`` through ``end_layer_idx`` (both inclusive), then are
scattered back into the dense token sequence.
"""

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.positional_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(dtype=t.dtype))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding), hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device)
                < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qk_norm=False,
        fused_attn=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
        )
        if hasattr(self.attn, "fused_attn"):
            self.attn.fused_attn = bool(fused_attn)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


@dataclass(frozen=True)
class Route:
    start_layer_idx: int
    end_layer_idx: int


def _normalize_routes(routes, depth):
    normalized = []
    for route in routes or []:
        if isinstance(route, Route):
            item = route
        else:
            item = Route(
                start_layer_idx=int(route["start_layer_idx"]),
                end_layer_idx=int(route["end_layer_idx"]),
            )
        if not 0 <= item.start_layer_idx <= item.end_layer_idx < depth:
            raise ValueError(
                "Each route must satisfy 0 <= start_layer_idx <= "
                f"end_layer_idx < {depth}; got {item}"
            )
        normalized.append(item)
    for previous, current in zip(normalized, normalized[1:]):
        if current.start_layer_idx <= previous.end_layer_idx:
            raise ValueError("TREAD routes must be ordered and non-overlapping")
    return tuple(normalized)


class TREADSiT(nn.Module):
    def __init__(
        self,
        path_type="linear",
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=None,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=True,
        z_dims=(768,),
        projector_dim=2048,
        qk_norm=False,
        fused_attn=False,
        routes=None,
    ):
        super().__init__()
        decoder_hidden_size = decoder_hidden_size or hidden_size
        if decoder_hidden_size != hidden_size:
            raise ValueError(
                "This inference implementation requires decoder_hidden_size "
                "to equal hidden_size, as in the official SiT-B/2 model."
            )
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.z_dims = tuple(int(value) for value in z_dims)
        self.routes = _normalize_routes(routes, depth)

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    fused_attn=fused_attn,
                )
                for _ in range(depth)
            ]
        )
        self.projectors = nn.ModuleList(
            [
                build_mlp(hidden_size, projector_dim, z_dim)
                for z_dim in self.z_dims
            ]
        )
        self.final_layer = FinalLayer(
            decoder_hidden_size, patch_size, self.out_channels
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        weight = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(weight.view(weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        channels = self.out_channels
        patch = self.x_embedder.patch_size[0]
        height = width = int(x.shape[1] ** 0.5)
        if height * width != x.shape[1]:
            raise ValueError("Token count must form a square grid")
        x = x.reshape(
            x.shape[0], height, width, patch, patch, channels
        )
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(
            x.shape[0], channels, height * patch, width * patch
        )

    @staticmethod
    def _gather_tokens(x, ids_keep):
        return x.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )

    @staticmethod
    def _restore_tokens(masked_x, ids_keep, original_x):
        return original_x.scatter(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, original_x.shape[-1]),
            masked_x,
        )

    def forward(
        self,
        x,
        t,
        y,
        force_routing=False,
        routing_ids_keep=None,
        return_projector_features=False,
    ):
        x = self.x_embedder(x) + self.pos_embed
        batch, _, hidden = x.shape
        condition = self.t_embedder(t) + self.y_embedder(y, self.training)
        projectors_output = []

        if force_routing:
            if len(self.routes) != 1:
                raise ValueError(
                    "Inference routing currently expects exactly one route"
                )
            if routing_ids_keep is None:
                raise ValueError(
                    "routing_ids_keep is required when force_routing=True"
                )
            ids_keep = routing_ids_keep.to(device=x.device, dtype=torch.long)
            if ids_keep.ndim != 2 or ids_keep.shape[0] != batch:
                raise ValueError(
                    "routing_ids_keep must have shape [batch, kept_tokens]"
                )
            if ids_keep.shape[1] < 1 or ids_keep.shape[1] > x.shape[1]:
                raise ValueError("Invalid number of kept routing tokens")
            if ids_keep.min() < 0 or ids_keep.max() >= x.shape[1]:
                raise ValueError("routing_ids_keep contains an invalid token index")
            route = self.routes[0]
        else:
            ids_keep = None
            route = None

        dense_tokens = None
        for index, block in enumerate(self.blocks):
            if route is not None and index == route.start_layer_idx:
                dense_tokens = x.clone()
                x = self._gather_tokens(x, ids_keep)

            x = block(x, condition)

            if route is not None and index == route.end_layer_idx:
                x = self._restore_tokens(x, ids_keep, dense_tokens)
                dense_tokens = None

            if return_projector_features and (index + 1) == self.encoder_depth:
                projectors_output = [
                    projector(x.reshape(-1, hidden)).reshape(
                        batch, x.shape[1], -1
                    )
                    for projector in self.projectors
                ]

        if dense_tokens is not None:
            raise RuntimeError("Routing sequence was not restored")
        velocity = self.unpatchify(self.final_layer(x, condition))
        return velocity, projectors_output


MODEL_CONFIGS = {
    "SiT-S/2": dict(depth=12, hidden_size=384, patch_size=2, num_heads=6),
    "SiT-B/2": dict(depth=12, hidden_size=768, patch_size=2, num_heads=12),
    "SiT-L/2": dict(depth=24, hidden_size=1024, patch_size=2, num_heads=16),
    "SiT-XL/2": dict(depth=28, hidden_size=1152, patch_size=2, num_heads=16),
}


def create_model(model_name="SiT-B/2", **kwargs):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported model {model_name!r}; choose from {list(MODEL_CONFIGS)}"
        )
    config = dict(MODEL_CONFIGS[model_name])
    config["decoder_hidden_size"] = config["hidden_size"]
    config.update(kwargs)
    return TREADSiT(**config)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    if embed_dim % 2:
        raise ValueError("Embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    positions = positions.reshape(-1)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def infer_projector_spec(state_dict: Mapping[str, torch.Tensor]):
    """Infer REPA projector dimensions from an official-style state dict."""
    indices = set()
    for key in state_dict:
        if key.startswith("projectors."):
            parts = key.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    if not indices:
        return [], 2048
    expected = list(range(max(indices) + 1))
    if sorted(indices) != expected:
        raise ValueError(f"Non-contiguous projector indices: {sorted(indices)}")
    z_dims = []
    projector_dim = None
    for index in expected:
        first = state_dict.get(f"projectors.{index}.0.weight")
        last = state_dict.get(f"projectors.{index}.4.weight")
        if first is None or last is None:
            raise ValueError(
                f"Cannot infer projector {index}: expected layers 0 and 4"
            )
        current_dim = int(first.shape[0])
        if projector_dim is None:
            projector_dim = current_dim
        elif projector_dim != current_dim:
            raise ValueError("Projectors use inconsistent hidden dimensions")
        z_dims.append(int(last.shape[0]))
    return z_dims, projector_dim

