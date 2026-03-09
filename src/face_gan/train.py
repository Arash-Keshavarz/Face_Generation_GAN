from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import DatasetDirectory, get_transforms
from .losses import discriminator_loss, generator_loss, gradient_penalty
from .models import Discriminator, Generator
from .utils import (
    default_run_name,
    device_from_string,
    ensure_dir,
    save_image_grid,
    set_seed,
    write_json,
)


def create_optimizers(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    lr: float = 1e-4,
) -> Tuple[optim.Adam, optim.Adam]:
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    return g_optimizer, d_optimizer


def generator_step(
    *,
    batch_size: int,
    latent_dim: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    g_optimizer: optim.Adam,
    device: torch.device,
) -> Dict[str, float]:
    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake_images = generator(noise)

    g_optimizer.zero_grad(set_to_none=True)
    g_loss = generator_loss(discriminator(fake_images))
    g_loss.backward()
    g_optimizer.step()
    return {"loss": float(g_loss.item())}


def discriminator_step(
    *,
    batch_size: int,
    latent_dim: int,
    real_images: torch.Tensor,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    d_optimizer: optim.Adam,
    device: torch.device,
    gp_weight: float = 1.0,
) -> Dict[str, float]:
    real_images = real_images.to(device)
    d_optimizer.zero_grad(set_to_none=True)

    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fake_images = generator(noise).detach()

    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images)
    d_loss = discriminator_loss(real_logits, fake_logits)
    gp = gradient_penalty(discriminator, real_images, fake_images)

    total_loss = d_loss + gp_weight * gp
    total_loss.backward()
    d_optimizer.step()

    return {"loss": float(d_loss.item()), "gp": float(gp.item())}


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str
    image_size: int = 64
    extension: str = ".jpg"
    latent_dim: int = 128
    conv_dim: int = 32
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    d_steps: int = 5
    g_steps: int = 1
    gp_weight: float = 1.0
    seed: int = 42
    deterministic: bool = False
    num_workers: int = 0
    device: str = "auto"
    log_every: int = 50
    sample_grid: int = 16
    run_name: str | None = None
    out_dir: str = "runs"
    resume: str | None = None


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    g_optimizer: optim.Adam,
    d_optimizer: optim.Adam,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(config),
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    g_optimizer: optim.Adam | None = None,
    d_optimizer: optim.Adam | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    if g_optimizer is not None:
        g_optimizer.load_state_dict(ckpt["g_optimizer_state_dict"])
    if d_optimizer is not None:
        d_optimizer.load_state_dict(ckpt["d_optimizer_state_dict"])
    return ckpt


def train(config: TrainConfig) -> Path:
    set_seed(config.seed, deterministic=config.deterministic)
    device = device_from_string(config.device)

    run_name = config.run_name or default_run_name("wgan_gp")
    run_dir = ensure_dir(Path(config.out_dir) / run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    samples_dir = ensure_dir(run_dir / "samples")

    write_json(run_dir / "config.json", asdict(config))

    dataset = DatasetDirectory(
        config.data_dir,
        transforms=get_transforms((config.image_size, config.image_size)),
        extension=config.extension,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        num_workers=config.num_workers,
    )

    generator = Generator(config.latent_dim, conv_dim=config.conv_dim).to(device)
    discriminator = Discriminator(conv_dim=config.conv_dim).to(device)
    g_optimizer, d_optimizer = create_optimizers(generator, discriminator, lr=config.lr)

    start_epoch = 0
    global_step = 0
    if config.resume:
        ckpt = load_checkpoint(
            config.resume,
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))

    fixed_latent = torch.randn(config.sample_grid, config.latent_dim, 1, 1, device=device)

    metrics_path = run_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "epoch",
                "batch",
                "d_loss",
                "d_gp",
                "g_loss",
            ],
        )
        if new_file:
            writer.writeheader()

        for epoch in range(start_epoch, config.epochs):
            for batch_i, real_images in enumerate(dataloader):
                real_images = real_images.to(device)

                for _ in range(config.d_steps):
                    d_metrics = discriminator_step(
                        batch_size=config.batch_size,
                        latent_dim=config.latent_dim,
                        real_images=real_images,
                        generator=generator,
                        discriminator=discriminator,
                        d_optimizer=d_optimizer,
                        device=device,
                        gp_weight=config.gp_weight,
                    )

                for _ in range(config.g_steps):
                    g_metrics = generator_step(
                        batch_size=config.batch_size,
                        latent_dim=config.latent_dim,
                        generator=generator,
                        discriminator=discriminator,
                        g_optimizer=g_optimizer,
                        device=device,
                    )

                if global_step % config.log_every == 0:
                    row = {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "batch": batch_i,
                        "d_loss": d_metrics["loss"],
                        "d_gp": d_metrics["gp"],
                        "g_loss": g_metrics["loss"],
                    }
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"epoch {epoch+1}/{config.epochs} | batch {batch_i}/{len(dataloader)} "
                        f"| d_loss {row['d_loss']:.4f} | gp {row['d_gp']:.4f} | g_loss {row['g_loss']:.4f}"
                    )

                global_step += 1

            generator.eval()
            with torch.no_grad():
                generated = generator(fixed_latent)
            save_image_grid(generated, samples_dir / f"epoch_{epoch+1:03d}.png", nrow=8)
            generator.train()

            ckpt_path = checkpoints_dir / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(
                ckpt_path,
                epoch=epoch + 1,
                global_step=global_step,
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                config=config,
            )
            save_checkpoint(
                checkpoints_dir / "latest.pt",
                epoch=epoch + 1,
                global_step=global_step,
                generator=generator,
                discriminator=discriminator,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                config=config,
            )

    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train WGAN-GP for face generation.")
    p.add_argument("--data-dir", default="processed_celeba_small/celeba/", help="Directory of training images.")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--extension", default=".jpg")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--conv-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-steps", type=int, default=5)
    p.add_argument("--g-steps", type=int, default=1)
    p.add_argument("--gp-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps|cuda:0 etc.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sample-grid", type=int, default=16)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        extension=args.extension,
        latent_dim=args.latent_dim,
        conv_dim=args.conv_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_steps=args.d_steps,
        g_steps=args.g_steps,
        gp_weight=args.gp_weight,
        seed=args.seed,
        deterministic=args.deterministic,
        num_workers=args.num_workers,
        device=args.device,
        log_every=args.log_every,
        sample_grid=args.sample_grid,
        run_name=args.run_name,
        out_dir=args.out_dir,
        resume=args.resume,
    )
    run_dir = train(config)
    print(f"done. artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()

