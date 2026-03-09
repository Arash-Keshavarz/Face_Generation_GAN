from __future__ import annotations

from typing import Dict

import torch


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = -real_logits.mean()
    fake_loss = fake_logits.mean()
    return real_loss + fake_loss


def gradient_penalty(
    discriminator: torch.nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty, implemented to match the notebook logic.
    """

    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
    x_hat = alpha * real_samples + (1 - alpha) * fake_samples
    x_hat.requires_grad = True

    pred = discriminator(x_hat)

    gradients = torch.autograd.grad(
        outputs=pred,
        inputs=x_hat,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


@torch.no_grad()
def step_metrics_to_log(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}

