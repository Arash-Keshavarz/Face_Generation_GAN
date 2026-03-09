# Portfolio Improvement Plan — Face Generation GAN

This file is a practical roadmap to turn this repository into a strong CV/portfolio project.

## 1) Make the project reproducible (highest impact)

- Add `requirements.txt` / `environment.yml` with pinned versions (PyTorch is platform-specific; document install).
- Add a quick-start section with exact commands:
  - environment setup
  - dataset placement / download instructions
  - training command
  - inference command
- Add deterministic seeds and mention reproducibility caveats for GPU/CUDA.

## 2) Convert notebook workflow into a small Python package

- Move model and training logic from `Face_Generation.ipynb` into `src/face_gan/`:
  - `src/face_gan/models.py`
  - `src/face_gan/data.py`
  - `src/face_gan/train.py`
  - `src/face_gan/generate.py`
- Keep the notebook for exploration/visualization only.

## 3) Add measurable evaluation

- Track at least one generative metric:
  - **FID** (preferred)
  - **IS** (optional)
- Report metric values by epoch in README as a small table.
- Save example generated grids at multiple checkpoints (e.g., 1, 5, 10, 20, 50).

## 4) Improve README storytelling for recruiters

- Add a short “Why this project matters” paragraph.
- Add model architecture + hyperparameters section.
- Add “What I tried / what failed / what improved” section.
- Add “Key engineering decisions” section (WGAN-GP choice, normalization, optimizer settings).
- Add a “Limitations & Ethical Considerations” section.

## 5) Show engineering maturity

- Add experiment tracking (TensorBoard screenshots, or W&B/MLflow if preferred).
- Save checkpoints and config files (`runs/<run>/config.json`, `runs/<run>/checkpoints/*.pt`).
- Add simple unit tests (transforms, denormalization, shape checks).
- Add CI (GitHub Actions): lint + test.

## 6) Add model cards and responsible AI notes

- Create `MODEL_CARD.md`:
  - intended use
  - non-intended use
  - training data summary
  - ethical risks and mitigations
- Add licensing/data usage notes for CelebA.

## 7) Prepare CV-ready evidence

- Add a “Results Summary” table:
  - baseline vs final model
  - training time
  - best checkpoint
  - metric gains
- Add a concise “Skills demonstrated” list:
  - PyTorch, GAN stabilization, data pipelines, evaluation, experiment hygiene.
- Add 2–3 bullet points that can be copied directly into a CV.

## 8) Stretch goals (if you have time)

- Compare WGAN-GP with a DCGAN baseline.
- Add latent space interpolation demo (GIF/video).
- Optional: deploy an inference demo app (Streamlit/Gradio).

## Suggested implementation order

1. Reproducibility + README quick-start.
2. Refactor notebook into scripts.
3. Add evaluation metrics + checkpointed outputs.
4. Add tests + CI + model card.
5. Add optional demo app.

