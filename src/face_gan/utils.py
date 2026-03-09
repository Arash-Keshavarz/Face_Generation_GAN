from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.utils import save_image


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def denormalize_to_uint8(images: np.ndarray) -> np.ndarray:
    """
    Transform images from [-1, 1] to [0, 255] as uint8 (matches notebook helper).
    """

    return ((images + 1.0) / 2.0 * 255.0).astype(np.uint8)


def default_run_name(prefix: str = "wgan_gp") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_image_grid(
    images: torch.Tensor,
    out_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, out_path, nrow=nrow, normalize=normalize, value_range=value_range)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")


def device_from_string(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)

