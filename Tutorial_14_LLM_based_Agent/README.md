# Tutorial 14 LLM-Based Agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_14_LLM_based_Agent/tutorial_14_pytorch.ipynb)

**Verified on Colab CPU (2026-10-15):** All nine code cells, controller tests, configuration/tokenizer/trace reloads, and evidence replay passed; four figures were inspected. The real model scored **4/8** on held-out routing and end-to-end questions. Failure cases are retained and explained; workflow execution is not a reliability claim.

Open in Colab, connect, and choose **Runtime → Run all**. Setup installs missing packages and downloads a public model. No local environment, API key, paid inference service, or Drive mount is needed.

## What you will learn

- Run Qwen2.5-0.5B-Instruct with PyTorch and compare a generated answer with an RDKit calculation.
- Inspect tokens and constrained next-token scores for tool routing.
- Validate requests, calculate HBD/HBA/molecular weight, and record action–observation evidence.
- Recover from molecule-name input using an explicit catalogue and a three-tool-call budget.
- Separate deterministic fault-injection checks from real LLM evaluation.
- Select between two prompts on eight development questions, then evaluate eight held-out questions.
- Save and reload configuration, tokenizer, and auditable traces.

## Resources and limits

The pinned public model downloads roughly 1 GB; CPU float32 weights occupy about 2 GB, plus runtime overhead. First-run download and inference can take several minutes. The six-molecule catalogue is embedded; no chemistry web API is needed. No model training is performed.

This is a constrained LLM router with a programmed controller, not an autonomous planner. Tool outputs are computed descriptors, not materials-performance predictions. The upstream paid GPT/LiteLLM and DSPy/GSM8K examples are replaced with a no-key local-model teaching workflow; DSPy and GSM8K are not implemented or verified here. GPU and longer-generation mode remain untested.

Output files go to `outputs/tutorial_14/`. Download them before Colab disconnects. See [validation evidence](../docs/validation.md).

License information is available in the repository LICENSE. The model and dependencies retain their own licenses.
