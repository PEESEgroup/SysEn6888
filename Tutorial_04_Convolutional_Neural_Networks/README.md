# Tutorial 04 Convolutional Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_04_Convolutional_Neural_Networks/tutorial_04_cnn.ipynb)

MNIST dense/CNN comparison and CIFAR-10 regularization and augmentation.

**Runtime:** CPU quick mode; GPU optional. Open the notebook, connect, and select **Runtime → Run all**. Dependency setup is included. Each notebook runs independently without local files or earlier lessons.

Basic PyTorch training (Tutorial 03) is helpful. Quick mode trains two models on MNIST and three prespecified variants on CIFAR-10 for three epochs each. Data augmentation affects only training images. Results are saved in `outputs/tutorial_04/metrics.json`; short-run comparisons do not establish the best architecture or regularization strategy.

See the [series README](../README.md) and [validation report](../docs/validation.md).

See the notebook for data references and the repository LICENSE for license information.


### First-run download time

**Allow extra time for the first CIFAR-10 download.** The full archive (~163 MB) downloads even with `QUICK_RUN=True`. Depending on the dataset server and network, this can take many minutes; one reported first run of this tutorial took about 30 minutes, largely during download. This is an observation, not an expected or guaranteed duration. A slow download does not necessarily mean the notebook is frozen: watch the download progress in the cell output and avoid repeatedly restarting while it advances. After downloading, extraction can take additional time without changing the progress bar. If an error appears, or progress remains unchanged for several minutes, check the runtime connection and retry the download cell. Rerunning in the same runtime normally reuses downloaded files; a new runtime may need to download them again.
