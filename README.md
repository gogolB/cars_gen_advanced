# Project: Advanced Generative Models for CARS Image Synthesis

This project focuses on developing and evaluating advanced generative models for synthesizing Coherent Anti-Stokes Raman Scattering (CARS) microscopy images of healthy and cancerous thyroid tissue. The primary goal is to generate high-quality synthetic images with realistic stromal patterns, especially given a small dataset.

## Approaches Explored:

This project will investigate several state-of-the-art generative modeling approaches in phases. Detailed documentation for each approach can be found in the `docs/approaches/` directory:

1.  [StyleGAN2-ADA / StyleGAN3](./docs/approaches/01_stylegan.md) (Initial Focus)
2.  [Projected GANs](./docs/approaches/02_projected_gan.md) (Planned)
3.  [Consistency Models](./docs/approaches/03_consistency_models.md) (Planned)

## Directory Structure:

- `configs/`: Hydra configuration files.
- `data/`: Raw and processed image data.
- `docs/`: Project documentation, including detailed READMEs for each modeling approach.
- `notebooks/`: Jupyter notebooks for experimentation and analysis.
- `outputs/`: Default Hydra directory for training outputs (models, logs, samples).
- `scripts/`: Utility scripts (data validation, generation, etc.).
- `src/`: Source code (DataModules, model definitions, Lightning modules, utilities).
- `tests/`: Unit and integration tests.
- `requirements.txt`: Python package dependencies.

## Setup & Usage:

(To be detailed as the project progresses)
