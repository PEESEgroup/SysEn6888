# Tutorial 03 Regression and Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_03_PyTorch_Regression_and_Classification/tutorial_03_pytorch.ipynb)

Build an Auto MPG regressor and a Fashion-MNIST classifier with PyTorch. Every required setup step is included in [the notebook](tutorial_03_pytorch.ipynb).

## What you need

- Basic Python knowledge, a browser, and internet access.
- CPU is enough; CUDA is selected automatically if available. Set `DEVICE_PREFERENCE="cpu"` to force CPU.
- No API key, paid service, or Drive mount.

## Two execution modes

| Setting | Regression epochs per model | Fashion-MNIST train / validation / test | Classification epochs |
|---|---:|---:|---:|
| `QUICK_RUN=True` (default) | 150 | 6,000 / 1,000 / 2,000 | 5 |
| `QUICK_RUN=False` | 300 | 54,000 / 6,000 / 10,000 | 10 |

Both modes use the same 274 / 58 / 60 Auto MPG split. Fashion-MNIST subsets are class-balanced and seeded. Both modes download the same approximately 30 MB of compressed Fashion-MNIST files. Mode changes require rerunning the notebook from the top.

The last section writes `metrics.json`, `mpg_regressor.pt`, and `fashion_classifier.pt` into `outputs/tutorial_03/` in the runtime. In Colab, download these from the Files sidebar before disconnecting.

## Optional local use for maintainers

Colab learners can ignore this section. For local testing, use a separate Python environment (the prepared environment uses Python 3.12). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt
python scripts/validate_tutorial03.py --execute --cpu
```

The validation script is provided to execute all cells in order in a fresh Jupyter kernel. It saves an executed copy, metrics, and plot images under `.validation/tutorial03/`, leaving the published notebook clean. See [the report](../docs/validation.md) for the actual validation status. To edit interactively, install JupyterLab in the environment and open the notebook.

## Changes from the upstream lesson

- Replaced scattered installation instructions and old torch/torchvision pins with a missing-package setup cell.
- Fixed the normalization path: fitted training statistics are actually applied to every split.
- Added a validation split, validation-selected checkpoints, and simple baselines.
- Kept test data out of preprocessing and model selection.
- Used sample-weighted epoch losses and separate model/optimizer state.
- Added CPU/CUDA device handling, seeded small-data mode, verified HTTPS downloads, error analysis, exercises, and preprocessing-aware checkpoints.
- Corrected explanations about one-hot encoding, standardization, and classification probabilities.

See the notebook for dataset and documentation references.
