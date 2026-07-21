import argparse
from pathlib import Path

import torch

from .model import model_from_checkpoint_args


def load_checkpoint(path, device, weights="ema"):
    path = Path(path)
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if weights not in ("ema", "model"):
        raise ValueError("weights must be 'ema' or 'model'")
    args = checkpoint["args"]
    state = checkpoint[weights]
    model = model_from_checkpoint_args(args, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # timm MLP versions may persist no-op norm parameters in other checkpoints.
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return model.eval().to(device), args, checkpoint.get("steps")
