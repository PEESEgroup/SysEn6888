# Colab validation report

Tutorials 01–06 were validated on **2026-10-09**, Tutorial 07 on **2026-10-10**, Tutorials 08–10 on **2026-10-11**, Tutorials 11–12 on **2026-10-12**, Tutorial 13 on **2026-10-13**, Tutorial 14 on **2026-10-15**, and Tutorial 15 on **2026-10-18**, using Google Colab CPU runtimes. The initial releases passed sequential execution (171 code cells total). The Tutorial 02 expansion below increases the current total to 186; other tutorials were not rerun for that expansion. Published notebooks have cleared outputs and execution counts.

## Tutorial 02 expansion: full notebook Run all verified (2026-10-26)

The complete 51-cell notebook was uploaded from the local release file to Colab and **Run all** was selected in a newly allocated CPU runtime. All 24 code cells completed in order, with execution counts 1–24 and the final `TUTORIAL_02_COMPLETE` marker. This is a full notebook execution, superseding the initial Tutorial 02 harness-only check. The code SHA-256 is recorded in the JSON report. A later Markdown-only note explains the unmatched county identifiers; tested code was unchanged.

The expanded material covers array construction, attributes, indexing, assignment, numerical edge cases and linear algebra; Series alignment, DataFrame editing and MultiIndex; Matplotlib axes, color palettes, line/marker styles and text; Seaborn relational/distribution/categorical plots; and a Plotly county choropleth. Intentional indexing exceptions were caught. The existing energy audit still produced 10 rows and 222 kWh. The tips download contained 244 records; the map matched 3,217 of 3,219 rate rows. Identifiers `02158` and `46102` did not match the supplied geometry and were explicitly reported, not silently remapped. All 17 expected exports were nonempty.

Twelve static figures and both interactive Plotly figures were visually inspected, including actual map rendering and county/energy hover labels. Download URLs and SHA-256 hashes are retained in the run metrics. Python 3.13.16, NumPy 2.1.3, Pandas 2.2.3, Matplotlib 3.10.0, Seaborn 0.13.2, and Plotly 5.24.1 were present. The missing-package installation branch was not exercised. External dataset URLs remain mutable; the hashes identify the tested downloads.

The browser automation interface timed out when downloading the executed notebook. Evidence consists of the uploaded local file, browser-visible sequential execution counts, captured output/metrics, and rendered figures; no downloaded-notebook equality check is claimed. This test used **File → Upload notebook**, not the post-publication GitHub badge. Published outputs are cleared. Existing local run records in Tutorials 01 and 12 were preserved and excluded from this release.

## How the checks were performed

At the time of the initial checks, the repository was private and the browser's Colab session was not authorized for GitHub access, the exact notebook code-cell sources were read from the repository and placed into Colab verification notebooks. A harness executed each source in order using Colab's native IPython `run_cell`, stopping on any exception. SHA-256 hashes of the code cells joined with two newlines identify the tested sources in [the machine-readable record](validation-results.json).

Tutorial 03 ran separately. Tutorials 01, 02, and 04 ran sequentially in a shared runtime; Tutorial 05 was then verified separately after fixing its data source. This verifies actual code execution in Colab. It is not a claim that the private GitHub authorization flow or an imported five-notebook **Run all** workflow was tested. No GitHub authorization or repository visibility change was performed.

The initial generic IPython-shell harness failed at `%matplotlib inline`; using the native Colab shell resolved this harness issue. The original Google-hosted cats/dogs URL returned HTTP 403. Tutorial 05 now uses the PyTorch-hosted ants/bees teaching archive, and the revised default code initially passed all 11 cells. The later ViT expansion increased Tutorial 05 to 16 code cells.

## Tutorial 05 ViT expansion: direct Run all verification

The expanded notebook was opened directly from GitHub commit `e0fde9f23c7c8c9b622e30fc16d08f8f47078d3e` in Colab and run with **Runtime → Run all** in a fresh CPU runtime. All 16 code cells completed without errors. The downloaded executed notebook matched that commit's code sources exactly. After inspecting the four rendered figures, the patch-sequence figure height/layout was adjusted to separate its titles, and Markdown approximation signs were clarified. These presentation-only adjustments were not rerun; both executed and published code hashes are recorded in the JSON. This later check supersedes the earlier harness-only limitation for Tutorial 05; it does not imply direct-opening tests for the other four lessons.

The expanded section compares the input image, numbered patch grid, patch sequence, pixel-vector and embedding heatmaps, positional embeddings, and untrained attention. Checks passed for exact image reconstruction (maximum error 0), convolution/linear projection equivalence, model forward equivalence, and attention row sums. Shapes were `[1,16,192]` for patches, `[1,16,64]` for projected tokens, and `[1,17,64]` after CLS/positions. The default full-notebook run took 35.15 seconds after imports on this runtime. The optional style branch was not rerun because its code was unchanged; its earlier evidence remains recorded.

## Tutorial 06: CPU execution verified

After the user freed Colab sessions, all 14 exact published code-cell sources executed sequentially in a fresh CPU runtime with Colab’s native `run_cell` harness. The downloaded verification notebook confirms exact source equality, no error outputs, and `ALL_14_CELLS_PASSED`. Four rendered figures (classes, learning curves, test comparison/confusion matrix, and predictions) were visually inspected. Direct GitHub opening required authorization and browser upload was unavailable; this check validates code execution, not the direct-opening or full notebook import workflow. No permissions or repository visibility were changed.

The flower archive and pretrained weights downloaded successfully. All assertions passed: disjoint splits, frozen backbone parameters and buffers, preserved head at unfreezing, unchanged protected fine-tuning state, changed unfrozen weights, cached/direct prediction equivalence, and CPU checkpoint reload. Quick mode used 200/50/50 images, 20 head epochs, and two fine-tuning epochs. Validation loss selected head epoch 11 and fine-tuning epoch 1. Trainable parameters increased from 6,405 to 892,485.

Frozen-feature validation/test accuracy was 82%/90%; fine-tuned accuracy was 84%/88%, against a 20% test baseline. Validation selected the fine-tuned model; its lower test accuracy is retained without retuning on the test set. With only 50 test images, the two-percentage-point difference is one image, not evidence of a general ranking. Fine-tuning is not guaranteed to improve test accuracy. ImageNet pretraining overlap was not audited.

Observed time after imports was 75.34 seconds, including 13.50 seconds caching, 0.35 seconds head fitting, and 35.32 seconds fine-tuning. Timings are observations, not guarantees or energy measurements. Full mode and GPU execution remain untested.

## Tutorial 07: detection, segmentation, and training verified

All 11 exact published code-cell sources ran sequentially in a fresh Colab CPU runtime using the native `run_cell` harness. The downloaded verification notebook was checked against the published sources, contained no error outputs, and ended with `ALL_11_CELLS_PASSED`. This checks real Colab execution, including installation of pinned **Ultralytics 8.4.158**, but does not establish direct GitHub authorization or full notebook import. Its Python/PyTorch environment matches the versions listed below.

The run downloaded YOLO11 nano detection and segmentation weights (5.4/5.9 MB) and COCO8. At image size 320, confidence 0.35, and NMS IoU 0.5, the bundled bus image produced four detections and four instance masks of shape `[4,1080,810]`. Coordinate/score bounds, original-resolution mask alignment, empty-result handling, unchanged background pixels, hand-computed IoU/NMS, and CSV/NPZ round trips all passed. Confidence cutoffs 0.2/0.5/0.8 yielded 5/4/3 instances at both tested NMS cutoffs; the synthetic overlapping-box example separately verified actual NMS suppression.

Default `RUN_TRAINING=True` completed two epochs on four training and four validation images. Label audits and train/validation hash separation passed; training losses were finite; `best.pt` reloaded and produced valid detection output. Five figures were inspected: input/boxes/masks, binary mask, synthetic NMS, ground-truth annotations, and training losses. COCO8 comes from COCO training data, so these are pipeline checks, not independent generalization results; reported validation mAP is retained only as diagnostic evidence.

The observed run took about 36 seconds including setup/imports; the notebook measured 18.28 seconds after imports and 10.22 seconds for training/checkpoint verification. These are not promised durations. A harmless config-directory warning caused Ultralytics to use `/tmp/Ultralytics`; execution and all exports succeeded. GPU, longer/custom-dataset training, segmentation training, and the optional training-skip branch were not separately tested.

## Tutorial 08 download repair (2026-10-26)

A reported `ConnectionRefusedError` / `URLError` occurred while connecting to the Stanford IMDb host. The download cell now tries the documented MindSpore-hosted archive before Stanford, with two attempts per source, a 30-second socket timeout, partial-file cleanup, and atomic replacement after SHA-256 verification. Both cached and new downloads must match the archive used in the earlier validation (`c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`). No extra framework dependency was added. Data selection, splits, model code, and training settings were unchanged.

The full notebook was uploaded to a newly allocated Colab CPU runtime and **Run all** completed all 11 cells. The mirror succeeded on its first attempt; archive verification, 5,994/998/1,995 splits, eight sentiment epochs, three Skip-gram epochs, TSV exports, padding checks, and CPU checkpoint reload all passed. Test accuracies remained 80.85% for masked embeddings and 84.21% for TF-IDF. Observed time after imports was 41.84 seconds, not a guarantee. Execution counts and final metrics were captured from browser output; no downloaded executed-notebook equality check is claimed.

Local mocked regression checks covered refused connections followed by fallback, valid-cache reuse without network access, corrupt-cache replacement, checksum-mismatch recovery, complete source failure, and partial-file cleanup. Live Stanford fallback, GPU, and larger mode were not tested in this run. The original validation below is retained as historical evidence.

## Tutorial 08: text classification and Skip-gram verified

All 11 exact published code-cell sources passed sequential execution using Colab's native `run_cell` harness on CPU. An initial fresh-runtime run downloaded the approximately 84 MB IMDb archive. After improving figure contrast and label placement, all cells were rerun in the same runtime with cached data. The final downloaded verification notebook matched the published sources exactly and contained no error outputs. Five rendered figures were inspected. This validates code execution, not direct private-GitHub authorization or full notebook import.

Exact-text deduplication yielded 5,994/998/1,995 training/validation/test reviews. Training-only vocabulary and IDF, one-hot/embedding equivalence, padding invariance, empty input, disjoint paths, finite losses, word/vector TSV alignment, and CPU checkpoint reload checks passed. The masked-embedding classifier completed eight epochs; validation loss selected epoch 5. Test accuracy was 80.85% versus 84.21% for TF-IDF and 49.97% for the training-majority baseline. Validation log loss selected the neural model, even though TF-IDF had higher accuracy; these are different criteria.

Skip-gram trained on 40,000 positive pairs sampled from 1,000 training reviews, using five noise samples and three epochs. Mean loss fell from 3.7249 to 2.4941. Many nearest neighbors remain poor: loss reduction and PCA plots do not demonstrate useful semantic representations. TSV exports contained 7,998 sentiment vectors and 4,194 observed Skip-gram target vectors. Full-data mode and GPU execution were not tested.

## Tutorial 09: encoder–decoder Transformer verified

All 13 exact published code-cell sources ran sequentially in a fresh Colab CPU runtime using the native `run_cell` harness, including setup, download, six training epochs, held-out greedy generation, exports, and checkpoint reload. The downloaded executed notebook matched the published code exactly, contained no error outputs, and ended with `ALL_13_CELLS_PASSED`. Five figures were inspected: token IDs/padding, causal attention, sinusoidal positions, learning curves, and trained attention heads. Direct private-GitHub authorization and full notebook import were not tested.

The 2.6 MB Google-hosted ManyThings/Tatoeba snapshot downloaded successfully. Filtering retained 111,422 short pairs; deterministic deduplication retained 92,421 pairs with unique normalized source and target sentences. The quick run used 3,000/400/400 train/validation/test pairs, source/target vocabularies of 3,000/2,651 tokens, and a 617,691-parameter model with one encoder block, one decoder block, and four attention heads. Both language sides were disjoint across splits; vocabularies used only training text.

Unicode preservation, shifted targets, manual/PyTorch attention equivalence, future-token invariance, appended-padding invariance, masked attention, finite training, scripted EOS termination, token-masked loss, CPU checkpoint logits, and greedy-generation equality all passed. Validation loss fell from 8.0975 before training to 4.4105 at the selected epoch 6. Test teacher-forced loss was 4.4576 and token accuracy 33.91%, versus a 12.5% constant-EOS baseline; known lexical-token accuracy was 26.52%. Source/target test OOV rates were 14.48%/7.75%.

Free-running generation reached EOS within the cap for 93.75% of the 400 test sentences, but strict normalized-reference exact match was **0%**. The first eight fixed examples were retained rather than selecting favorable translations. This verifies the training/inference implementation, not useful translation quality. Single-reference exact match is strict, and teacher-forced token accuracy must not be described as translation accuracy. Observed time after imports was 48.21 seconds; it is not a promised duration. GPU and larger mode remain untested.

## Tutorial 10: AE, denoising, ECG, VAE, and AAE verified

All 13 exact published code-cell sources passed in a fresh Colab CPU runtime with the native `run_cell` harness, including automatic downloads, five model training workflows, and CPU checkpoint reloads. The final version was rerun after adding inline display of the fixed-prior GIF. The downloaded executed notebook matched the published sources exactly, contained no error outputs, and ended with `ALL_13_CELLS_PASSED`. Ten figures and the first/middle/last frames of the inline seven-frame GIF were inspected. Observed time after imports was 34.68 seconds, not a promised duration. This checks actual code execution, not direct private-GitHub authorization or full notebook import. The default `RUN_AAE=True` branch was included; skip, GPU, and larger modes were not separately tested.

Fashion-MNIST used 4,000/800/1,000 train/validation/test images. Both AE and denoiser selected epoch 4. The dense AE's held-out MSE was 0.03738 versus 0.08739 for the training-mean-image baseline. Denoising reduced MSE from 0.05247 for the unchanged noisy input to 0.03250. Architectures differ, so their scores are not a controlled architecture comparison.

The ECG CSV contained 4,998 unique feature rows, split 2,998/1,000/1,000. The AE fit 1,751 normal training rows, selected epoch 50 using normal validation MAE, and calibrated its threshold on 584 normal validation rows. Anomaly means CSV label **0**, and the 95th-percentile threshold was 0.04760. Test accuracy/precision/recall/F1 were 94.8%/95.05%/92.31%/93.66%, ROC AUC 0.9813, and false-positive rate 3.42%. Confusion counts were TN=564, FP=20, FN=32, TP=384. This is a row-level demonstration; no patient-independent or clinical performance claim is supported.

The 2D convolutional VAE used 6,000/1,000/1,000 statically binarized MNIST images and selected epoch 6. Its seeded test objective was 188.1716 nats/image = 182.9207 reconstruction + 5.2509 KL, with beta=1. Analytic KL, zero-KL, reparameterization gradients, shape/range, objective units, fixed-epsilon CPU reload, and fixed-prior decoder equality checks passed. The seven-frame GIF retained the same prior codes from initialization through six epochs. Probability images, Bernoulli samples, posterior means, and interpolation were explicitly distinguished. Outputs remain blurry in this small run.

The AAE trained on 2,000 images for two epochs and selected epoch 2 by validation reconstruction. Snapshot assertions verified shared-encoder updates and isolation of reconstruction, discriminator, and encoder-adversarial phases. Its prior samples remained unrecognizable; passing these checks establishes the update mechanism, not successful distribution matching. All five model checkpoints reloaded on CPU, and ECG CSV export and GIF frame-count checks passed.

## Tutorial 11: GAN generation and translation verified

All 11 exact published code-cell sources passed sequential execution in Colab CPU using native `run_cell`. The initial fresh runtime exercised automatic MNIST downloads. After improving the GAN generator, the final version was rerun in a restarted Python session using the cached dataset. The downloaded executed notebook matched all published code sources exactly, contained no error outputs, and ended with `ALL_11_CELLS_PASSED`. Twelve figures and initialization/middle/final frames of the 13-frame inline GIF were visually inspected. A subsequent Markdown-only addition describes these observed results; code was unchanged. No direct private-GitHub authorization or full-notebook import was tested. GPU and larger mode remain untested.

The final configuration used 6,000/1,000/1,000 MNIST train/validation/test images, 12 DCGAN epochs, four paired epochs on 2,000 images, and four unpaired epochs with 1,000 disjoint source rows per domain. GAN and CycleGAN inference snapshots use the final epoch; the paired generator uses validation-selected epoch 4. These are simplified MNIST teaching architectures, not facades or horse/zebra reproductions.

Checks verified discriminator/generator update isolation (including GAN batch-normalization buffers), detached fake images, gradients through the frozen discriminator, latent-only inversion with an unchanged generator, paired/unpaired update isolation, disjoint splits/domain pools, finite losses, logit/shape conventions, CSV/GIF round trips, and **all eight network CPU reloads**. Inference snapshots omit optimizer/RNG state and do not promise exact training resumption. Time after imports was 104.84 seconds on this runtime; this is not a promised duration.

Fixed-noise samples had more varied but still fragmented digit-like strokes; no FID or class-coverage measure was computed. On one fixed held-out example, latent inversion reduced MSE from **0.3515 to 0.1159**. Paired test L1 was **0.1017**, compared with 0.3083 for the input-edge baseline and 0.3009 for the training-mean baseline. The patch discriminator scores were near-uniform; these comparisons do not isolate the adversarial term's benefit.

The unpaired demonstration did not achieve clean contrast inversion: X→Y/Y→X target L1 was **1.2543/1.0198**, while cycle L1 was **0.1813/0.1365**. Checkerboard artifacts and incorrect contrast remained visible. Returning the input has target L1 1.8911; analytic contrast inversion has zero error by construction. This illustrates that a relatively small cycle error does not establish the intended translation. All errors above use the [-1,1] pixel scale; they are instructional diagnostics, not benchmark claims.

## Tutorial 12: discrete and continuous reinforcement learning verified

All 10 exact published code-cell sources executed sequentially in a fresh Colab CPU runtime using native `run_cell`. The downloaded executed notebook matched the published code exactly, contained no error outputs, and ended with `ALL_10_CELLS_PASSED`. Seven figures and initialization/middle/final frames of two policy GIFs were inspected. Explanatory Markdown was subsequently added with these observed results; code did not change. Direct private-GitHub authorization/full notebook import, GPU, larger budgets, and repeated training seeds were not tested.

Runtime used **Gymnasium 1.3.0**, Python 3.13.16, PyTorch 2.11.0+cpu, NumPy 2.1.3, Pandas 2.2.3, and Matplotlib 3.10.0. Default training budgets were 240 CartPole episodes (seed 42), 6,000 Pendulum steps (seed 43; 2,750 updates), and 4,000 materials steps (seed 44; 1,750 updates). Each policy used evaluation seeds 10000–10009; evaluation did not select checkpoints. Total time after imports was **55.00 seconds** on this runtime, not a promised duration.

| Task | Trained policy return, mean ± sample SD | Random return, mean ± sample SD | Other baseline |
|---|---:|---:|---|
| CartPole-v1 | 352.8 ± 52.9 | 25.8 ± 14.1 | Initial greedy: 9.2 ± 0.8 |
| Pendulum-v1 | −682.3 ± 230.9 | −1057.0 ± 251.9 | Zero torque: −1071.7 ± 325.0; initial actor: −1082.3 ± 307.4 |
| Synthetic materials | −1.939 ± 0.577 | −26.422 ± 12.233 | No adjustment: −23.687 ± 11.321; analytic: −0.695 ± 0.613 |

These statistics describe episode variability for one trained agent per task, not robustness across training seeds. CartPole improved substantially without establishing a solved-task claim. Pendulum control remained imperfect. The known-target analytic materials controller outperformed DDPG; this example demonstrates constrained numerical control, not real materials discovery. The materials trajectory approached a slightly offset composition while preserving the simplex.

Termination versus truncation targets, detached actor advantages, replay wraparound/ending flags, DDPG parameter/gradient isolation, frozen targets and actor-target soft-update arithmetic, action bounds, finite losses, the Gymnasium custom-environment checker, and simplex invariants passed. All **nine network checkpoints reloaded on CPU** with matching probe outputs. The evaluation CSV contained 110 rows. Files are inference/inspection snapshots, not exact optimizer/replay/RNG resume states.

RGB rendering worked without Xvfb. The first predetermined seed produced an 85-frame CartPole GIF (334 steps) and a 51-frame Pendulum GIF (200 steps); these are separate illustrative rollouts, not extra evaluation episodes. Pygame emitted a non-fatal `pkg_resources` deprecation warning; rendering and GIF readback passed. No Atari/DQN or PPO implementation is claimed: the upstream PyTorch lesson only listed them in its introductory outline, and this adaptation provides a conceptual comparison.

## Tutorial 13: graph learning and molecular regression verified

All 10 exact published code-cell sources executed sequentially in a fresh Colab CPU runtime using native `run_cell`. The downloaded executed notebook matched every code source, had no error outputs, and ended with `ALL_10_CELLS_PASSED`. This verifies actual Colab execution, including package setup, automatic Cora/IMDB/QM9 downloads, model fitting, and six CPU checkpoint round trips. It does not verify private-GitHub authorization or full notebook import. Code was unchanged after execution; only explanatory Markdown was added.

Environment: Python 3.13.16, PyTorch 2.11.0+cpu, PyG 2.8.1, RDKit 2026.09.1, NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0. Quick mode used seed 42 on CPU. Runtime after imports was 46.33 seconds; this excludes setup and is not a duration guarantee.

Manual neighbor sums (including an isolated node), disconnected mini-batches, GCN/dense equivalence, GraphSAGE/reference equivalence, finite gradients, node permutation equivariance, graph pooling invariance, batched/separate predictions, split separation, training-only normalization, and six checkpoint reloads all passed. Six rendered figures were inspected for content and layout: messages/batch adjacency, Cora curves/accuracy/confusion, IMDB curves/accuracy, molecular drawings, QM9 distributions, and regression parity/baselines.

Cora used the public 140/500/1,000 node masks. After 120 epochs per model, validation-selected checkpoints yielded test accuracies **58.8% MLP, 81.2% GCN, and 80.8% GraphSAGE**. Validation loss selected GraphSAGE; its slightly lower test accuracy did not trigger reselection. The training-majority rule has a seven-way tie (20 labels per class); class ID 0 wins `argmax`, producing a 13% test baseline. This is a transductive experiment and does not demonstrate unseen-graph generalization.

IMDB-BINARY used 700/150/150 stratified graphs and 35 epochs per model: **69.33% GIN-0 and 70.67% GIN-eps** test accuracy, versus 50% majority. Sum pooling caused large initial loss spikes before convergence. Graph IDs are disjoint, but isomorphic graphs across splits were not audited. One split/seed is not the upstream repeated cross-validation benchmark.

QM9 used a roughly 30 MB DeepChem CSV and 3,000 seeded unique canonical SMILES (2,100/450/450); no invalid or duplicate SMILES were encountered in this selected subset. Only molecular structure enters features. Dipole target scaling was fitted on training molecules. The 45-epoch GCN selected epoch 40 by validation MAE (0.7745 D). Test MAE was **0.7633 D**, versus **1.0381 D** for validation-selected atom-count Ridge and **1.1965 D** for the training mean. The parity plot shows underestimation of rare high-dipole molecules. SMILES-derived graphs omit bond types and 3D geometry; random splits can share scaffolds. This is not a crystal-property, new-chemistry, or materials-discovery benchmark. GPU and larger mode remain untested.

## Recorded environment and results

Python **3.13.16**, PyTorch **2.11.0+cpu**, torchvision **0.26.0+cpu**, NumPy **2.1.3**, Pandas **2.2.3**, Matplotlib **3.10.0**. Tutorial 02 also used Seaborn **0.13.2** and Plotly **5.24.1**. Machine-learning runs used seed 42 and `QUICK_RUN=True`.

| Tutorial | Code cells | Observed result |
|---|---:|---|
| 01 | 8 | Five synthetic readings: total 55 kWh, mean 11 kWh; JSON save/reload passed. |
| 02 | 9 | 13 raw rows → 10 clean rows; total 222 kWh; CSV round trip and figure/HTML exports passed. |
| 03 | 17 | Validation-selected MLP: test MAE 1.917 MPG, RMSE 2.638 MPG (baseline RMSE 7.432). Fashion-MNIST test accuracy 81.75% (baseline 10%). Both checkpoint reload checks passed. |
| 04 | 10 | MNIST CNN test accuracy 92.2%; dense model 88.4%. Validation-selected CIFAR-10 Plain CNN: test accuracy 44.65% (baseline 10.2%). All five model runs completed. |
| 05 | 16 | Ants/bees: validation accuracy 95%, test accuracy 93.33% on 60 test images (baseline 50%); both-class Grad-CAM, model reload, and ViT shape checks passed. |
| 06 | 14 | Frozen features: test accuracy 90%; fine-tuned: 88% on 50 images (baseline 20%). All freezing, fine-tuning, and checkpoint assertions passed. |
| 07 | 11 | Four detections/four masks; five figures inspected; IoU/NMS, coordinate/mask, empty-result, export, two-epoch training, and checkpoint inference checks passed. |
| 08 | 11 | IMDb neural/TF-IDF test accuracy 80.85%/84.21%; three Skip-gram epochs; five figures, padding, TSV, and checkpoint checks passed. |
| 09 | 13 | Six Transformer epochs; test teacher-forced token accuracy 33.91%, greedy exact match 0%; causal/padding/EOS/CPU reload checks and five figures passed. |
| 10 | 13 | Five model reloads; denoising MSE 0.03250 versus 0.05247 noisy input; VAE objective 188.17 nats/image; AAE update-isolation checks, ten figures, and seven-frame GIF passed. |
| 11 | 11 | Twelve GAN epochs; latent inversion MSE 0.1159; paired test L1 0.1017; unpaired translation limitations recorded; eight reloads, 12 figures, and 13-frame GIF checked. |
| 12 | 10 | CartPole mean return 352.8; Pendulum −682.3; synthetic materials −1.939; termination/replay/update checks, nine reloads, seven plots, and two policy GIFs passed. |
| 13 | 10 | Cora GCN/GraphSAGE 81.2%/80.8%; IMDB GIN 69.3%/70.7%; QM9 MAE 0.7633 D; six figures and six checkpoint reloads checked. |
| 14 | 9 | Real local Qwen model; 4/8 held-out accuracy with errors retained; tool/controller/reload checks passed; four figures inspected. |
| 15 | 9 | Dense/TF-IDF Hit@1 7/8 and 6/8; generation content probe 1/4, valid citation IDs 0/4, exact abstention 1/2; index checks and four figures inspected. |

These are observed teaching runs, not benchmark guarantees. In Tutorial 04, three epochs did not make the augmented model outperform the plain model. In Tutorial 05, the archive is an ImageNet subset and the feature extractor was pretrained on ImageNet: these results do not demonstrate independence from pretraining data or generalization to a new domain.

CIFAR-10 downloading dominated Tutorial 04's approximately 22.6-minute recorded run; server speed varies. Quick mode reduces training, not the size of the source archive download. Tutorial 05's core run took about 29 seconds after imports on this runtime; this is not a promised duration.

The optional Tutorial 05 style-transfer branch also passed on CPU with `RUN_STYLE_TRANSFER=True`: 128 × 128 pixels, 40 optimization steps, finite losses throughout, and a loss decrease from 454,754.125 to 35,324.152. This branch was run separately after the core lesson. The saved Grad-CAM and style-transfer images were displayed in Colab and visually inspected.

## Tutorial 14: local LLM agents verified, accuracy limits retained

All nine exact code-cell sources completed sequentially in a fresh Colab CPU runtime using native `run_cell`. Rendered output was saved locally with matching source SHA-256 and `ALL_9_CELLS_PASSED`. The browser's notebook download did not produce an accessible file; therefore, unlike previous runs, no downloaded executed-notebook equality check is claimed. Four PNGs were extracted from the rendered output DOM and inspected: molecular structures/descriptors, action scores, the live recovery trace, and development/test accuracy. Published code was unchanged after execution; explanatory Markdown was added.

Environment: Python 3.13.16, PyTorch 2.11.0+cpu, Transformers 5.18.0, RDKit 2026.09.1, NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0. The public Qwen2.5-0.5B-Instruct checkpoint at `7ae557604adf67be50417f59c2c2f167def9a775` downloaded without a token; 494,032,768 parameters were loaded. The unauthenticated Hub warning and `torch_dtype` deprecation warning were nonfatal. The run used 27 actual routing forward passes and one direct generated answer; 93.66 seconds elapsed after imports, including model download/loading.

The direct model answer for aspirin's HBD count was **2**; RDKit returned **1**. The live agent selected HBD and completed the three-call invalid-SMILES → name resolution → descriptor recovery path. Tool schema/value/error checks, six controller fault checks, bounded traces, disjoint development/test wording, configuration and tokenizer reloads, trace JSON round trip, and descriptor evidence replay passed. Recovery is programmed controller behavior; test fixtures are explicitly separated from real LLM evaluations. This is not a model-weight reload test.

Two fixed prompts scored **4/8** and **6/8** on development questions. The latter was selected without test feedback; held-out routing and end-to-end accuracy were both **4/8 (50%)**. Donor/acceptor paraphrases were misrouted to molecular weight, and both unsupported questions received unrelated descriptors. These errors are retained without tuning on the held-out set. Successful descriptor calculation does not establish correct intent recognition or reliable abstention. The notebook explains RDKit-specific descriptor conventions, including standalone water's 0/0 counts in these functions.

No paid API, DSPy/GSM8K experiment, external chemistry resolver, GPU execution, longer-generation mode, private-GitHub import, or autonomous laboratory behavior is claimed. The small catalogue, two prompts, and hand-written cases are teaching examples, not broad reliability evidence.

## Tutorial 15: RAG execution verified, generation failures retained

All nine exact code-cell sources completed sequentially in a fresh Colab CPU runtime with native `run_cell`. Rendered output contained matching SHA-256 `17d55538ec67b6e65113205a44b8fa3f15e14773a76f041b93cfca2354b16fe8` and `ALL_9_CELLS_PASSED`. Four rendered PNGs were extracted and inspected: chunk boundaries/lengths, embedding similarities, retrieval comparison, and generation diagnostics. No downloaded executed-notebook equality or direct private-GitHub import check is claimed. Only explanatory Markdown was added after execution.

Environment: Python 3.13.16, PyTorch 2.11.0+cpu, Transformers 5.18.0, NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0. Default CPU quick mode downloaded the pinned MiniLM and Qwen public checkpoints without a token. The Hub unauthenticated-access and `torch_dtype` deprecation warnings were nonfatal. Nine actual generation calls completed; time after imports, including downloads, was 91.99 seconds, not a duration guarantee.

Twelve documents (eleven embedded notes/fictional materials records and one live article) produced 97 chunks and a 97 × 384 dense index. Source text SHA-256 was `c0ffaa16be6c28865a87fa896593249f9067a02b57ee87d528acd4262bbc08bf`. One chunk was truncated at the embedding tokenizer's 256-token limit; full text remained in the stored chunk. Offset coverage, unique IDs, normalized embeddings, padding equivalence, metadata filters, empty-result/no-model-call policy, JSON/vector/manifest round trips, ranking equality, and modified-chunk detection passed. These checks do not establish semantic correctness or model-weight reload.

Eight fixed retrieval questions yielded dense Hit@1 **7/8**, Hit@3 **8/8**, and mean reciprocal rank within three **0.9375**; TF-IDF yielded **6/8**, **8/8**, and **0.875**. Explicit kind filters were supplied by the evaluation harness; no automatic query analysis is claimed. Dense retrieval put the 300 K record first for the 100 K question. No parameters were retuned on these diagnostics.

The initial no-context answer was `NO_EVIDENCE`; RAG returned the correct fictional 300 K gap, 1.73 eV, but omitted a citation. Across four additional answerable questions, the content substring probe passed **1/4** and valid supplied citation IDs **0/4**. The generator invented 1.85 eV instead of the 100 K value 1.91 eV, returned `[17]` for a duration, and `[chunk_id] M5_0` for an atmosphere. Uncertainty 0.06 eV was reproduced correctly. Two missing-property questions produced **1/2 exact abstentions**; the other included `NO_EVIDENCE` after malformed citation text, so it attempted abstention but failed the strict format check. All raw outputs are retained in the machine-readable record. The blog answer has no automatic correctness label.

This is a pipeline teaching demonstration with a small local generator, not reliable scientific QA. Materials values are explicitly synthetic. No paid calls, LangGraph/LangSmith, asynchronous/token streaming, model query rewriting, GPU, longer-generation, or prompt-injection-resistance result is claimed. The live article and pretrained weights are fetched at runtime, not committed.

## Scope and remaining limits

- CPU quick-mode code execution was tested for Tutorials 01–15 (Tutorial 07 includes its default two-epoch smoke test). GPU execution and larger `QUICK_RUN=False` experiments were not tested.
- Tutorials 01–06 and 08–11 found their required packages already installed. Tutorial 07 exercised its pinned Ultralytics installation in Colab. Other missing-package branches were not exhaustively tested in stripped runtimes.
- Runtime data and trained weights are not committed. Notebook assertions check splits, shapes, finite values, or checkpoint equality where relevant.
- Local notebook execution was not performed after the initial sandbox kernel restriction. Local package installation is not counted as runtime validation.
- Colab resources, installed package versions, and upstream download availability can change. If a runtime has unrelated package modifications, start a fresh runtime before running the lesson.

## Reproduce

Open a lesson in Colab, connect to a fresh CPU runtime, and run cells in order. Save the final outputs and environment information, plus the JSON summary in `outputs/tutorial_XX/` (Tutorial 01 uses `energy_summary.json`; Tutorial 13 uses `run_summary.json`). Test full/GPU/optional modes separately and record their settings.

From a local environment containing `nbformat` and IPython, `python scripts/check_notebooks.py` checks all fifteen notebooks for schema, unique IDs, cleared outputs, Python/IPython syntax, and Colab targets without starting kernels. The optional `scripts/validate_tutorial03.py --execute --cpu` runner executes Tutorial 03 in a fresh local kernel and saves results under `.validation/`; this was not the method used for the Colab results above.
