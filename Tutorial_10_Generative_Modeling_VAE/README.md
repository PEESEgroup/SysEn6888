# Tutorial 10 Autoencoders and VAEs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_10_Generative_Modeling_VAE/tutorial_10_pytorch.ipynb)

Self-contained PyTorch reconstruction, denoising, ECG anomaly scoring, convolutional VAE generation, and a small adversarial autoencoder demonstration.

- Installs missing packages and downloads Fashion-MNIST, MNIST, and the ECG CSV automatically (~50 MB). No local environment, TensorFlow, API key, or Drive mount.
- Quick mode: Fashion 4,000/800/1,000 images; MNIST 6,000/1,000/1,000 images; four AE/denoising epochs, 50 ECG epochs, six VAE epochs, and two AAE epochs on 2,000 images.
- Ten figures show data, reconstruction/denoising, ECG results, VAE losses, prior generation, posterior/latent grids, interpolation, and AAE samples. A fixed-prior GIF shows training progress.
- Validation selects checkpoints and calibrates the ECG threshold. The test set does not select epochs or thresholds. ECG positive class is anomaly (CSV label 0).
- CPU reload checks cover all five models. VAE KL/reparameterization and AAE shared-encoder/update-isolation checks are included.
- GPU/larger mode and `RUN_AAE=False` require separate validation. ECG is a row-level educational example, not a clinical diagnostic system; quick generative runs are not quality benchmarks.

Outputs in `outputs/tutorial_10/` include figures, GIF, checkpoints, ECG predictions, and metrics.

[Series index](../README.md) · [Validation evidence](../docs/validation.md).
