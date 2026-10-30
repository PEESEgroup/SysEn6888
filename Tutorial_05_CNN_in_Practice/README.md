# Tutorial 05 CNNs in Practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_05_CNN_in_Practice/tutorial_05_cnn_practice.ipynb)

Transfer learning, feature maps, Grad-CAM, optional neural style transfer, and a visual ViT image-to-token walkthrough.

**Runtime:** CPU core lesson; optional style transfer is heavier. Open the notebook, connect, and select **Runtime → Run all**. Dependency setup is included. Each notebook runs independently without local files or earlier lessons.

Basic CNN concepts (Tutorial 04) are helpful. The core lesson trains a new classification head using frozen MobileNetV2 features, inspects filters and activations, computes Grad-CAM, and verifies a saved model. Outputs go to `outputs/tutorial_05/`. Grad-CAM is a model diagnostic, not a causal explanation.

**Optional style transfer:** `RUN_STYLE_TRANSFER = False` by default. Enabling it downloads about 550 MB of VGG19 weights; quick mode optimizes a 128-pixel image for 40 steps. The final tiny ViT uses random weights to show image grids, patch sequences, pixel-vector and embedding heatmaps, CLS/positional embeddings, and attention. It verifies exact patch reconstruction and matching convolution/linear projections. Four `vit_*.png` figures and a patch-size comparison table make the intermediate representations visible; these are not trained predictions.

See the [series README](../README.md) and [validation report](../docs/validation.md).

See the notebook for data references and the repository LICENSE for license information.
