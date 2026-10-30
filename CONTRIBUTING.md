# Maintaining the Colab tutorial series

Tutorial 03 is the template for the PyTorch and framework-independent lessons.

Each released notebook should:

1. State its question, prerequisites, learning goals, runtime requirements, and any costs.
2. Include a Colab badge, all imports, dependency setup when needed, and automatic public-data loading.
3. Run from a fresh runtime in cell order without local files, hidden state, personal credentials, or earlier notebooks.
4. Provide a small default example and explicitly label heavier optional experiments.
5. Explain tensor shapes, losses, units, baselines, and limitations of results.
6. Fit preprocessing only on training data and use validation data for model selection.
7. Include interpretation prompts and exercises that do not break Run all.
8. Record actual validation evidence, versions, device, and tested modes. Local execution must not be labeled as a live Colab test.
9. Preserve upstream attribution and dataset terms; never commit API keys, downloaded datasets, or large checkpoints.

Run `python scripts/check_notebooks.py` for structure, syntax, output hygiene, and Colab link checks. It requires `nbformat` and IPython and does not execute notebooks. For runtime checks, use a fresh Colab runtime and run cells in order; record metrics and inspect plots, then update `docs/validation.md`. Check heavier optional branches separately and state any untested modes. Keep published outputs cleared.

An additional local execution runner for Tutorial 03 is available with `python scripts/validate_tutorial03.py --execute --cpu`. It starts a fresh local kernel, downloads data, and saves an executed notebook and plots under `.validation/`. Dependencies in `requirements-validation.txt` describe a prepared local environment, not the tested Colab environment. Notebook setup intentionally retains installed packages unless a required package is missing.
