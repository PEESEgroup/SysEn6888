# Tutorial 15 Retrieval-Augmented Generation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_15_Retrieval_Augmented_Generation/tutorial_15_pytorch.ipynb)

**Verified on Colab CPU (2026-10-18):** All nine code cells and index checks passed; four figures were inspected. Dense/TF-IDF Hit@1 was 7/8 and 6/8, with Hit@3 of 8/8 for both. Generation remained weak: content probes 1/4, valid citation IDs 0/4, exact abstention 1/2. Raw failures and their interpretation are retained.

Open in Colab, connect to a CPU runtime, and choose **Runtime → Run all**. Setup installs missing packages, fetches a web article, and downloads public model weights. No local environment, API key, paid inference service, or Drive mount is required.

## What you will learn

- Load a real article and preserve source IDs, metadata, offsets, and hashes.
- Visualize chunk overlap and embedding similarity.
- Compare TF-IDF search with 384-dimensional MiniLM embeddings implemented in PyTorch.
- Apply explicit metadata filters and inspect top-k evidence.
- Compare local Qwen answers with and without retrieval.
- Evaluate retrieval, answer content probes, citation IDs, and abstention separately.
- Inspect pipeline step events and save/reload a dense index with a provenance manifest.

## Resources and scope

Public MiniLM and Qwen checkpoints download about 1.1 GB in total. CPU float32 inference needs several GB of RAM; download and execution time vary. `QUICK_RUN=False` increases generation length. No training is performed.

The materials records are **fictional teaching data**, not scientific measurements. The article is downloaded from [Lilian Weng's site](https://lilianweng.github.io/posts/2023-06-23-agent/) at runtime and is not committed. Source changes can alter retrieval; the run records its content hash.

This adaptation replaces paid upstream API calls with local models and uses plain Python for the retrieval/generation flow. LangGraph, LangSmith, token streaming, asynchronous execution, and LLM query rewriting are outside this lesson. A valid citation ID does not establish factual support; the small model can ignore evidence or fail to abstain. GPU and longer mode need separate validation.

Download `outputs/tutorial_15/` before disconnecting. It contains plots, chunk metadata, vectors, the manifest, raw answers, diagnostic CSVs, and `run_summary.json`. Source-document and model licenses remain separate from the code license. See [validation evidence](../docs/validation.md).

License information is available in the repository LICENSE.
