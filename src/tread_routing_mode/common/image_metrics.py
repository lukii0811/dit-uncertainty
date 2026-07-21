"""DINOv2 embedding similarity and LPIPS distance for paired trajectories."""

from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class ImageMetricComparator:
    """Compute the two image-space metrics used by all routing experiments."""

    def __init__(self, dino_model="facebook/dinov2-base", device="cpu", batch_size=8,
                 lpips_net="alex"):
        from transformers import AutoImageProcessor, AutoModel
        import lpips

        project_root = Path(__file__).resolve().parents[3]
        huggingface_cache = project_root / "weights" / "huggingface" / "hub"
        local_dino_cache = (
            huggingface_cache / f"models--{dino_model.replace('/', '--')}"
        ).exists()
        self.dino_model = dino_model
        self.device = device
        self.batch_size = batch_size
        self.lpips_net = lpips_net
        self.processor = AutoImageProcessor.from_pretrained(
            dino_model, cache_dir=huggingface_cache, use_fast=False,
            local_files_only=local_dino_cache)
        self.model = AutoModel.from_pretrained(
            dino_model, cache_dir=huggingface_cache,
            use_safetensors=True,
            local_files_only=local_dino_cache).to(device).eval()

        torch.hub.set_dir(str(project_root / "weights" / "torch" / "hub"))
        self.lpips_model = lpips.LPIPS(
            net=lpips_net, version="0.1", verbose=False).to(device).eval()

        self.embedding_cache = {}

    @torch.inference_mode()
    def _embed_missing(self, paths):
        paths = [Path(path).resolve() for path in paths]
        missing = list(dict.fromkeys(
            path for path in paths if path not in self.embedding_cache
        ))
        for offset in range(0, len(missing), self.batch_size):
            batch_paths = missing[offset:offset + self.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            cls_embeddings = self.model(**inputs).last_hidden_state[:, 0].float().cpu()
            for path, embedding in zip(batch_paths, cls_embeddings):
                self.embedding_cache[path] = embedding

    def _load_pixels(self, path):
        with Image.open(Path(path).resolve()) as image:
            return pil_to_tensor(image.convert("RGB")).float().div(127.5).sub(1.0)

    def compare_paths(self, reference_paths, candidate_paths):
        if len(reference_paths) != len(candidate_paths):
            raise ValueError("reference_paths and candidate_paths must have equal lengths")
        self._embed_missing([*reference_paths, *candidate_paths])
        metrics = []
        for offset in range(0, len(reference_paths), self.batch_size):
            reference_batch_paths = reference_paths[offset:offset + self.batch_size]
            candidate_batch_paths = candidate_paths[offset:offset + self.batch_size]
            reference_pixels = torch.stack([
                self._load_pixels(path) for path in reference_batch_paths]).to(self.device)
            candidate_pixels = torch.stack([
                self._load_pixels(path) for path in candidate_batch_paths]).to(self.device)
            with torch.inference_mode():
                lpips_values = self.lpips_model(
                    reference_pixels, candidate_pixels).flatten().cpu()

            for reference_path, candidate_path, lpips_value in zip(
                    reference_batch_paths, candidate_batch_paths, lpips_values):
                reference_embedding = self.embedding_cache[Path(reference_path).resolve()]
                candidate_embedding = self.embedding_cache[Path(candidate_path).resolve()]
                dino_similarity = functional.cosine_similarity(
                    reference_embedding[None], candidate_embedding[None]).item()
                dino_similarity = max(-1.0, min(1.0, dino_similarity))
                metrics.append({
                    "dino_similarity": float(dino_similarity),
                    "lpips_distance": float(lpips_value.item()),
                })
        return metrics

    def compare_sample_dirs(self, reference_dir, candidate_dir):
        reference_images = {
            path.stem: path for path in (Path(reference_dir) / "images").glob("step_*.png")
        }
        candidate_images = {
            path.stem: path for path in (Path(candidate_dir) / "images").glob("step_*.png")
        }
        if reference_images.keys() != candidate_images.keys():
            missing = sorted(reference_images.keys() - candidate_images.keys())
            extra = sorted(candidate_images.keys() - reference_images.keys())
            raise ValueError(
                "Decoded trajectories do not contain identical steps: "
                f"missing={missing}, extra={extra}"
            )
        names = sorted(reference_images, reverse=True)
        if not names:
            raise ValueError("No matching decoded trajectory images were found")
        metrics = self.compare_paths(
            [reference_images[name] for name in names],
            [candidate_images[name] for name in names],
        )
        return [
            {
                "step": int(name.removeprefix("step_")),
                "dino_model": self.dino_model,
                "lpips_net": self.lpips_net,
                **row,
            }
            for name, row in zip(names, metrics)
        ]
