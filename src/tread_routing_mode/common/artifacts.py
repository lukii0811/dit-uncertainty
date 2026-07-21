import hashlib
import json
from pathlib import Path

from PIL import Image


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_image(tensor, path):
    image = ((tensor.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(path)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
