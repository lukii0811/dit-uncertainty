"""Checkpoint inspection and strict SiT model construction."""

from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import torch

try:
    from .sit_tread import MODEL_CONFIGS, create_model, infer_projector_spec
except ImportError:
    from sit_tread import MODEL_CONFIGS, create_model, infer_projector_spec


STATE_KEY_CANDIDATES = (
    "ema",
    "model",
    "state_dict",
    "ema_state_dict",
    "model_state_dict",
)


def load_checkpoint_file(path, allow_unsafe_pickle=False):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    try:
        # PyTorch 2.6 defaults to weights_only=True. Mentor checkpoints made
        # with argparse commonly store their training arguments as a Namespace
        # next to the tensor state dict. Namespace is passive metadata, so
        # allowlist only that class while retaining the safer weights-only
        # unpickler instead of immediately falling back to arbitrary pickle.
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            return torch.load(path, map_location="cpu", weights_only=True)
        with safe_globals([Namespace]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception as error:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "Safe checkpoint loading failed. If this is a trusted mentor "
                "checkpoint containing Python metadata, rerun with "
                "--allow-unsafe-checkpoint-load. Original error: "
                f"{error}"
            ) from error
        return torch.load(path, map_location="cpu", weights_only=False)


def _is_tensor_state_dict(value):
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def extract_state_dict(checkpoint, state_key="auto"):
    if state_key != "auto":
        if not isinstance(checkpoint, Mapping) or state_key not in checkpoint:
            available = list(checkpoint) if isinstance(checkpoint, Mapping) else []
            raise KeyError(
                f"Checkpoint has no state key {state_key!r}; available: {available}"
            )
        state_dict = checkpoint[state_key]
        if not _is_tensor_state_dict(state_dict):
            raise TypeError(f"Checkpoint entry {state_key!r} is not a state dict")
        return state_dict, state_key

    if _is_tensor_state_dict(checkpoint):
        return checkpoint, "<root>"
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping or a tensor state dict")
    for candidate in STATE_KEY_CANDIDATES:
        if candidate in checkpoint and _is_tensor_state_dict(checkpoint[candidate]):
            return checkpoint[candidate], candidate
    tensor_mappings = [
        key for key, value in checkpoint.items() if _is_tensor_state_dict(value)
    ]
    if len(tensor_mappings) == 1:
        key = tensor_mappings[0]
        return checkpoint[key], key
    raise KeyError(
        "Could not identify a state dict automatically. Top-level keys: "
        f"{list(checkpoint)}; pass --state-key explicitly."
    )


def strip_state_prefixes(state_dict):
    cleaned = {}
    prefixes = ("module.", "_orig_mod.")
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        key = key.replace("._orig_mod.", ".")
        if key in cleaned:
            raise ValueError(f"State-key collision after prefix cleanup: {key}")
        cleaned[key] = value
    for container_prefix in ("model.", "net."):
        if cleaned and all(
            key.startswith(container_prefix) for key in cleaned
        ):
            cleaned = {
                key[len(container_prefix) :]: value
                for key, value in cleaned.items()
            }
            break
    return cleaned


def infer_backbone_summary(state_dict):
    patch_weight = state_dict.get("x_embedder.proj.weight")
    label_weight = state_dict.get("y_embedder.embedding_table.weight")
    if patch_weight is None or label_weight is None:
        raise KeyError(
            "Checkpoint does not look like an official-style SiT state dict: "
            "x_embedder.proj.weight or y_embedder.embedding_table.weight is missing"
        )
    hidden_size = int(patch_weight.shape[0])
    in_channels = int(patch_weight.shape[1])
    patch_size = int(patch_weight.shape[2])
    if patch_weight.shape[2] != patch_weight.shape[3]:
        raise ValueError("Non-square patch embedding is unsupported")
    block_indices = set()
    for key in state_dict:
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
    depth = max(block_indices) + 1 if block_indices else 0
    embedding_rows = int(label_weight.shape[0])
    num_classes = embedding_rows - 1
    return {
        "hidden_size": hidden_size,
        "in_channels": in_channels,
        "patch_size": patch_size,
        "depth": depth,
        "label_embedding_rows": embedding_rows,
        "num_classes": num_classes,
    }


def _validate_requested_model(model_name, summary):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model_name!r}")
    expected = MODEL_CONFIGS[model_name]
    checks = {
        "hidden_size": expected["hidden_size"],
        "patch_size": expected["patch_size"],
        "depth": expected["depth"],
    }
    mismatches = {
        key: (summary[key], value)
        for key, value in checks.items()
        if summary[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual}, requested={wanted}"
            for key, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"Checkpoint is not compatible with {model_name}: {details}")


def _filter_ignored_prefixes(state_dict, prefixes):
    prefixes = tuple(prefixes)
    if not prefixes:
        return state_dict, []
    kept = {}
    ignored = []
    for key, value in state_dict.items():
        if key.startswith(prefixes):
            ignored.append(key)
        else:
            kept[key] = value
    return kept, ignored


def build_model_from_checkpoint(
    checkpoint_path,
    model_name="SiT-B/2",
    state_key="auto",
    resolution=256,
    num_classes=1000,
    encoder_depth=8,
    start_layer_idx=2,
    end_layer_idx=8,
    qk_norm=False,
    fused_attn=True,
    ignored_state_prefixes=("projectors.",),
    allow_unsafe_pickle=False,
):
    checkpoint = load_checkpoint_file(
        checkpoint_path, allow_unsafe_pickle=allow_unsafe_pickle
    )
    raw_state, resolved_state_key = extract_state_dict(checkpoint, state_key)
    state_dict = strip_state_prefixes(raw_state)
    summary = infer_backbone_summary(state_dict)
    _validate_requested_model(model_name, summary)
    if summary["in_channels"] != 4:
        raise ValueError(
            f"Expected four latent channels, found {summary['in_channels']}"
        )
    if summary["num_classes"] != num_classes:
        raise ValueError(
            "Class embedding mismatch: checkpoint implies "
            f"{summary['num_classes']} classes plus one null class, but "
            f"--num-classes={num_classes}"
        )

    z_dims, projector_dim = infer_projector_spec(state_dict)
    ignore_projectors = any(
        prefix == "projectors." for prefix in ignored_state_prefixes
    )
    routes = [
        {
            "start_layer_idx": start_layer_idx,
            "end_layer_idx": end_layer_idx,
        }
    ]
    model = create_model(
        model_name,
        path_type="linear",
        input_size=resolution // 8,
        in_channels=summary["in_channels"],
        num_classes=num_classes,
        class_dropout_prob=0.1,
        encoder_depth=encoder_depth,
        # REPA projectors are training-only. Avoid allocating/moving them to
        # the GPU when their checkpoint prefix is intentionally ignored.
        z_dims=[] if ignore_projectors else z_dims,
        projector_dim=projector_dim,
        qk_norm=qk_norm,
        fused_attn=fused_attn,
        routes=routes,
    )

    loadable_state, ignored_checkpoint_keys = _filter_ignored_prefixes(
        state_dict, ignored_state_prefixes
    )
    incompatible = model.load_state_dict(loadable_state, strict=False)
    missing_nonignored = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(tuple(ignored_state_prefixes))
    ]
    unexpected_nonignored = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith(tuple(ignored_state_prefixes))
    ]
    if missing_nonignored or unexpected_nonignored:
        raise RuntimeError(
            "The Mentor checkpoint is not compatible with this SiT baseline. "
            f"Missing non-projector keys: {missing_nonignored[:30]}; "
            f"unexpected non-projector keys: {unexpected_nonignored[:30]}. "
            "Do not bypass this error: these parameters can affect generation."
        )

    report = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "state_key": resolved_state_key,
        "checkpoint_top_level_keys": (
            list(checkpoint) if isinstance(checkpoint, Mapping) else []
        ),
        "backbone": summary,
        "model_name": model_name,
        "resolution": resolution,
        "encoder_depth_assumed": encoder_depth,
        "routes": routes,
        "projector_z_dims_inferred": z_dims,
        "projector_hidden_dim_inferred": projector_dim,
        "projectors_instantiated": not ignore_projectors,
        "ignored_state_prefixes": list(ignored_state_prefixes),
        "ignored_checkpoint_key_count": len(ignored_checkpoint_keys),
        "ignored_checkpoint_keys_preview": ignored_checkpoint_keys[:20],
        "missing_ignored_model_keys": [
            key
            for key in incompatible.missing_keys
            if key.startswith(tuple(ignored_state_prefixes))
        ][:20],
        "strict_generative_backbone_match": True,
    }
    return model, report
