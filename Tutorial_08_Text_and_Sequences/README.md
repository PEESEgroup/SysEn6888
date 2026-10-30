# Tutorial 08 Text, Sentiment, and Word Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_08_Text_and_Sequences/tutorial_08_pytorch.ipynb)

Self-contained PyTorch text tutorial: token IDs and embedding heatmaps, masked-pooling IMDb sentiment classification, a TF-IDF baseline, and Skip-gram with negative sampling on many training reviews.

- Setup and IMDb download run inside Colab; no local environment, TensorFlow, torchtext, API key, or Drive mount.
- Default CPU-friendly mode: up to 6,000/1,000/2,000 train/validation/test reviews (exact duplicates removed), eight sentiment epochs, 40,000 positive pairs and three Skip-gram epochs.
- `QUICK_RUN=False`: up to 20,000/5,000/25,000 reviews, 15 sentiment epochs, up to 300,000 pairs and eight Skip-gram epochs. GPU is optional; larger/GPU modes require separate validation.
- IMDb downloads try the documented [MindSpore mirror](https://mindnlpdocs-traineval.readthedocs.io/en/latest/api/utils.html) and then Stanford, with bounded retries. Downloaded and cached archives must match the SHA-256 of the previously validated dataset; partial downloads are never extracted. No extra framework is installed.
- Full IMDb archive is about 84 MB even in quick mode. Vocabulary and IDF fit training text only; models are selected using validation loss.
- Outputs in `outputs/tutorial_08/`: five figures, aligned word/vector TSVs, a reload-checked neural checkpoint, and metrics.

Includes padding/empty-input checks, a hand-checkable skip-gram example, cosine neighbors and PCA interpretation limits. Small Word2Vec runs do not establish general semantic understanding. No data or model weights are redistributed.

[Series index](../README.md) · [Validation evidence](../docs/validation.md).
