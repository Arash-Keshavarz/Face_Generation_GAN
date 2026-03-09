import torch

from face_gan.models import Discriminator, Generator


def test_generator_output_shape() -> None:
    g = Generator(latent_dim=128, conv_dim=32)
    z = torch.randn(4, 128, 1, 1)
    x = g(z)
    assert x.shape == (4, 3, 64, 64)


def test_discriminator_output_shape() -> None:
    d = Discriminator(conv_dim=32)
    x = torch.randn(4, 3, 64, 64)
    y = d(x)
    assert y.shape == (4, 1, 1, 1)

