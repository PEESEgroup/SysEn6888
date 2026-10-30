# Tutorial 09 Encoder–Decoder Transformers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_09_Transformers_Implementation/tutorial_09_pytorch.ipynb)

Build a small PyTorch Spanish → English Transformer with explicit encoder/decoder blocks, sinusoidal positions, causal and padding masks, teacher forcing, and greedy decoding.

- Includes environment setup and automatic 2.6 MB ManyThings/Tatoeba snapshot download. No local installation, TensorFlow, torchtext, API key, or Drive mount.
- Quick mode: 3,000/400/400 short train/validation/test pairs, one encoder and decoder layer, dimension 64, four heads, and six epochs.
- Larger mode: 12,000/1,500/1,500 short pairs and 15 epochs. GPU and larger mode need separate validation.
- Five figures explain IDs/padding, causal attention, positions, learning curves, and trained attention heads.
- Checks cover future-token leakage, padding invariance, special tokens, EOS termination, masked loss, and CPU checkpoint reload.
- Teacher-forced accuracy is not translation accuracy. Held-out greedy examples and strict exact match expose free-running limitations.

The upstream lesson uses GRUs plus attention despite its title; this adaptation implements an actual Transformer. Small teaching runs are not production translators.

[Series index](../README.md) · [Validation evidence](../docs/validation.md).
