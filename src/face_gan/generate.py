from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models import Generator
from .utils import device_from_string, ensure_dir, save_image_grid, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate faces from a trained generator checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to a checkpoint (epoch_XXX.pt or latest.pt).")
    p.add_argument("--out", default="outputs/generated.png", help="Output image path for a grid.")
    p.add_argument("--num", type=int, default=64, help="Number of images to generate.")
    p.add_argument("--nrow", type=int, default=8, help="Grid nrow.")
    p.add_argument("--latent-dim", type=int, default=None, help="Override latent dim (otherwise read from checkpoint config).")
    p.add_argument("--conv-dim", type=int, default=None, help="Override conv dim (otherwise read from checkpoint config).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=123)
    return p


@torch.no_grad()
def generate(
    *,
    checkpoint: str | Path,
    out: str | Path,
    num: int,
    nrow: int,
    device: torch.device,
    seed: int,
    latent_dim: int | None = None,
    conv_dim: int | None = None,
) -> Path:
    set_seed(seed, deterministic=False)
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_cfg = ckpt.get("config", {}) or {}

    latent_dim = int(latent_dim or ckpt_cfg.get("latent_dim", 128))
    conv_dim = int(conv_dim or ckpt_cfg.get("conv_dim", 32))

    generator = Generator(latent_dim=latent_dim, conv_dim=conv_dim).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    noise = torch.randn(num, latent_dim, 1, 1, device=device)
    images = generator(noise)

    out_path = Path(out)
    ensure_dir(out_path.parent)
    save_image_grid(images, out_path, nrow=nrow)
    return out_path


def main() -> None:
    args = build_parser().parse_args()
    device = device_from_string(args.device)
    out_path = generate(
        checkpoint=args.checkpoint,
        out=args.out,
        num=args.num,
        nrow=args.nrow,
        device=device,
        seed=args.seed,
        latent_dim=args.latent_dim,
        conv_dim=args.conv_dim,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

