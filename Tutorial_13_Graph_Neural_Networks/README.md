# Tutorial 13 Graph Neural Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PEESEgroup/SysEn6888/blob/main/Tutorial_13_Graph_Neural_Networks/tutorial_13_pytorch.ipynb)

Open in Colab, connect to a runtime, and choose **Runtime → Run all**. Package installation and data downloads happen inside the notebook; no local environment setup is required.

## What you will build

- A visual trace of neighbor messages and disconnected graph mini-batches.
- GCN and mean GraphSAGE layers with dense/reference and permutation checks.
- Cora node classification: MLP, GCN, and GraphSAGE with validation-selected checkpoints.
- IMDB-BINARY graph classification: GIN-0 versus learned-epsilon GIN.
- Real QM9 dipole regression in Debye, with mean and atom-count Ridge baselines.
- Six CPU checkpoint round trips, metric CSVs, prediction exports, and a JSON run summary.

## Default workload and interpretation

`QUICK_RUN=True`: 120 epochs for each Cora model, 35 epochs for each GIN, and 45 QM9 epochs on 3,000 molecules (2,100/450/450). Cora uses its public masks; IMDB uses a stratified 700/150/150 split. Setup retains installed PyTorch and installs missing PyG/RDKit dependencies. The QM9 CSV is about 30 MB; the full 3D archive is not needed. All figures are generated in the notebook.

This is one seeded teaching experiment. Cora is transductive; IMDB graph IDs are disjoint but isomorphic graphs were not audited; QM9 uses random, unique-SMILES splits and omits 3D coordinates and bond types. These are not benchmark or materials-discovery claims. GPU and larger mode require separate verification.

Save files from `outputs/tutorial_13/` before Colab disconnects. See [validation evidence](../docs/validation.md) for actual runtime checks and their limits.

License information is available in the repository LICENSE. Datasets and dependencies retain their own terms.
