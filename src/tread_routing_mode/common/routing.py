"""Deterministic routing schedules for the baseline and TREAD experiments."""

import hashlib

import torch


class RoutingSchedule:
    MODES = {"baseline", "resample", "single_timestep"}

    def __init__(self, *, mode, keep_ratio, mask_seed, total_steps,
                 num_tokens, batch_size=1, route_step=None):
        if mode not in self.MODES:
            raise ValueError(f"Unknown routing mode: {mode}")
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if num_tokens <= 0 or batch_size <= 0:
            raise ValueError("num_tokens and batch_size must be positive")
        if mode == "single_timestep":
            if route_step is None or not 1 <= int(route_step) <= int(total_steps):
                raise ValueError("single_timestep mode requires route_step in [1, total_steps]")
        elif route_step is not None:
            raise ValueError("route_step is only valid in single_timestep mode")
        self.mode = mode
        self.keep_ratio = float(keep_ratio)
        self.mask_seed = int(mask_seed)
        self.total_steps = int(total_steps)
        self.num_tokens = int(num_tokens)
        self.batch_size = int(batch_size)
        self.route_step = int(route_step) if route_step is not None else None
        self.generator = torch.Generator(device="cpu").manual_seed(mask_seed)
        self.identity = torch.arange(num_tokens).expand(batch_size, -1).clone()
        self.draw_count = 0
        self.history = []

    def _sample_ids(self):
        self.draw_count += 1
        num_keep = max(1, int(self.num_tokens * self.keep_ratio))
        noise = torch.rand((self.batch_size, self.num_tokens), generator=self.generator)
        return noise.argsort(dim=1)[:, :num_keep]

    @staticmethod
    def _hash(ids):
        return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()

    def routing_for_step(self, step, device):
        step = int(step)
        if not 1 <= step <= self.total_steps:
            raise ValueError(f"step must be in [1, {self.total_steps}], got {step}")
        active = self.mode == "resample" or (
            self.mode == "single_timestep" and step == self.route_step
        )
        masking_active = active and self.keep_ratio < 1.0
        if not active:
            ids = None
        elif masking_active:
            ids = self._sample_ids()
        else:
            ids = self.identity
        self.history.append({
            "step": step,
            "t": float(step / self.total_steps),
            "routing_active": bool(active),
            "routing_selected": bool(active),
            "masking_active": bool(masking_active),
            "keep_ratio": self.keep_ratio if active else 1.0,
            "masked_ratio": 1.0 - self.keep_ratio if active else 0.0,
            "num_kept": int(ids.shape[1]) if active else self.num_tokens,
            "num_routed": int(self.num_tokens - ids.shape[1]) if active else 0,
            "mask_draw_index": self.draw_count if masking_active else None,
            "mask_sha256": self._hash(ids) if active else None,
        })
        return active, ids.to(device) if ids is not None else None
