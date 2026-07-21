import math

import torch
from torch import nn


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return self.mlp(torch.cat((args.cos(), args.sin()), dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels, unconditional=False):
        if unconditional:
            labels = torch.full_like(labels, self.num_classes)
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        inner = int(dim * ratio)
        self.fc1 = nn.Linear(dim, inner)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(inner, dim)
        self.drop2 = nn.Dropout(0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        sm, scm, gm, sf, scf, gf = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gm[:, None] * self.attn(_modulate(self.norm1(x), sm, scm))
        return x + gf[:, None] * self.mlp(_modulate(self.norm2(x), sf, scf))


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class PatchEmbed(nn.Module):
    def __init__(self, size, patch, channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SiT(nn.Module):
    """Checkpoint-compatible SiT with explicit TREAD routing controls."""
    def __init__(self, input_size, patch_size, channels, dim, depth, heads,
                 num_classes, route_start, route_end):
        super().__init__()
        self.in_channels = channels
        self.patch_size = patch_size
        self.route_start = route_start
        self.route_end = route_end
        self.x_embedder = PatchEmbed(input_size, patch_size, channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.y_embedder = LabelEmbedder(num_classes, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (input_size // patch_size) ** 2, dim), requires_grad=False)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.final_layer = FinalLayer(dim, patch_size, channels)

    def forward(self, x, t, y, *, unconditional=False, route=True,
                route_ids=None):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, unconditional)
        original = ids = None
        for index, block in enumerate(self.blocks):
            if route and index == self.route_start:
                original = x
                if route_ids is None:
                    raise ValueError("Explicit route_ids are required")
                ids = route_ids.to(x.device)
                if ids.ndim != 2 or ids.shape[0] != x.shape[0]:
                    raise ValueError("route_ids must have shape [batch, kept_tokens]")
                x = x.gather(1, ids[..., None].expand(-1, -1, x.shape[-1]))
            x = block(x, c)
            if route and index == self.route_end:
                x = original.scatter(1, ids[..., None].expand(-1, -1, x.shape[-1]), x)
        x = self.final_layer(x, c)
        b, n, d = x.shape
        p, ch, side = self.patch_size, self.in_channels, int(math.sqrt(n))
        x = x.reshape(b, side, side, p, p, ch)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(b, ch, side * p, side * p)


def model_from_checkpoint_args(args, state):
    model_name = args.model.replace("DiT", "SiT")
    specs = {
        "SiT-S/2": (384, 12, 6, 2), "SiT-B/2": (768, 12, 12, 2),
        "SiT-L/2": (1024, 24, 16, 2), "SiT-XL/2": (1152, 28, 16, 2),
    }
    if model_name not in specs:
        raise ValueError(f"Unsupported checkpoint architecture: {args.model}")
    dim, depth, heads, patch = specs[model_name]
    latent_side = int(math.sqrt(state["pos_embed"].shape[1])) * patch
    channels = state["x_embedder.proj.weight"].shape[1]
    return SiT(latent_side, patch, channels, dim, depth, heads, args.num_classes,
               args.start_layer_idx, args.end_layer_idx)
