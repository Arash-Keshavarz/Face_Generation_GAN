import torch

from face_gan.losses import discriminator_loss, generator_loss, gradient_penalty
from face_gan.models import Discriminator, Generator


def test_losses_return_scalars() -> None:
    g = Generator(latent_dim=128)
    d = Discriminator()
    z = torch.randn(2, 128, 1, 1)
    fake = g(z)
    real = torch.randn(2, 3, 64, 64)
    real_logits = d(real)
    fake_logits = d(fake)

    gl = generator_loss(fake_logits)
    dl = discriminator_loss(real_logits, fake_logits)
    gp = gradient_penalty(d, real, fake.detach())

    assert gl.ndim == 0
    assert dl.ndim == 0
    assert gp.ndim == 0

