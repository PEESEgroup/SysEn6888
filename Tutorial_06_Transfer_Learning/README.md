# Tutorial 06 Transfer Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_06_Transfer_Learning/tutorial_06_pytorch.ipynb)

Train a five-class flower classifier, then continue fine-tuning the final two MobileNetV2 feature blocks. This self-contained PyTorch notebook extends Tutorial 05's frozen-feature lesson with explicit parameter/buffer audits and a comparison of both strategies.

- Dependencies and data download are included. No local installation, API key, or Drive mount.
- CPU quick mode: 200 training / 50 validation / 50 test images; 20 head epochs and two fine-tuning epochs.
- Full mode: approximately 80/10/10 stratified split; 40 head epochs and five fine-tuning epochs. GPU is optional.
- Downloads: about 218 MB flower archive and 14 MB pretrained weights. Quick mode still downloads the full archive.
- Outputs: learning curves, test comparisons, predictions, `metrics.json`, and a saved `flower_classifier.pt` with a reload-and-inference example under `outputs/tutorial_06/`.

The notebook also explains residual connections, epochs versus steps, correct binary/multiclass decoding, pretrained ImageNet inference, and why `eval()` differs from parameter freezing. Fine-tuning keeps the learned head, uses smaller learning rates, recomputes features from images, and selects checkpoints using validation loss (including epoch zero).

[Series README](../README.md) · [Validation report](../docs/validation.md). Dataset terms are separate and retained in the download.
