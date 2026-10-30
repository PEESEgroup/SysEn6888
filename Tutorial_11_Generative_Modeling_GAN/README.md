# Tutorial 11 GANs: Generation and Image Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_11_Generative_Modeling_GAN/tutorial_11_pytorch.ipynb)

Run the notebook from the top in Colab. Dependencies install inside the runtime; no local environment, API key, Drive mount, or pretrained model is required.

## What you will learn

- A compact convolutional GAN with explicit discriminator/generator update checks.
- Fixed-noise samples and a training GIF, latent interpolation, and frozen-generator inversion.
- Paired edge-to-digit translation with a U-Net skip connection and patch discriminator.
- Unpaired ordinary/inverted digit translation with adversarial, cycle, and identity losses.
- Baselines, held-out diagnostics, CSV exports, and reload-and-inference examples for all eight saved networks.

Default quick mode trains on MNIST in a CPU-compatible workflow: twelve DCGAN epochs, four paired epochs, and four unpaired epochs. Download is about 12 MB. Results are short teaching runs, not quality or convergence guarantees. `QUICK_RUN=False` increases data and epochs; optional GPU selection is automatic.

## Scope

This reworks the upstream PyTorch GAN lesson into one independent notebook. Small MNIST translation tasks replace facades and horse/zebra datasets; networks and schedules are simplified. It is not a reproduction of pix2pix or CycleGAN benchmark results. The known inversion target is used only for final diagnostics; training pools for unpaired domains have disjoint source rows. Low adversarial/cycle loss does not prove realistic or semantically correct generation.

Outputs are written to `outputs/tutorial_11/`; download them before Colab disconnects. Model files are inference snapshots rather than exact resume checkpoints. See [validation evidence](../docs/validation.md) for tested modes and observed limitations.

License information is available in the repository LICENSE.
