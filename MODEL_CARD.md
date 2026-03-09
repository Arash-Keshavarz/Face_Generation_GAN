# Model Card — Face Generation GAN (WGAN-GP)

## Model details

- **Model type**: Generative Adversarial Network (Wasserstein GAN with Gradient Penalty)
- **Outputs**: 64×64 RGB face images
- **Implementation**: PyTorch (`src/face_gan/`)

## Intended use

- Educational / research exploration of GAN training stability (WGAN loss + gradient penalty)
- Portfolio demonstration of deep learning engineering and experimentation workflows

## Non-intended use

- Impersonation, deception, or generation of images representing real individuals in misleading contexts
- Any use that violates the dataset license/terms or local laws

## Training data

- **Dataset**: CelebA (Large-scale CelebFaces Attributes Dataset)
- **Preprocessing**: Resize to 64×64, normalize to [-1, 1]
- **Notes**: This repo expects images to be placed in a local folder (see `README.md`). The dataset itself is not included.

## Evaluation

- **Current**: qualitative sample grids saved during training
- **Planned**: FID tracking against a held-out subset of real images

## Limitations

- Low resolution (64×64) limits fine-grained detail
- Training can be unstable; results vary by seed/hardware
- Potential demographic imbalance or bias inherited from the dataset

## Ethical considerations

- Face generation can be misused; keep generated samples clearly labeled as synthetic
- Avoid deploying this model in contexts where identity or authenticity matters
- Respect dataset terms and privacy expectations of subjects in the original dataset

## How to use

- Train: `face-gan-train --data-dir <path>`
- Generate: `face-gan-generate --checkpoint runs/<run>/checkpoints/latest.pt`

