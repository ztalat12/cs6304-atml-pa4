# Task 1 — Model compression of Qwen2.5-1.5B (base)

Notebook: [`atml-pa4-task1.ipynb`](atml-pa4-task1.ipynb) (Kaggle: `atml-pa4-task1`).

| Run | Kaggle version | Notebook | Results |
|---|---|---|---|
| v1 — first full run | 2 | commit `1b1bf4f` (then named `atmt-pa4-task1.ipynb`) | [`results_v1/`](results_v1/) |
| v2 — per-group Q3 + memory fix | 3 | current [`atml-pa4-task1.ipynb`](atml-pa4-task1.ipynb) | `results_v2/` (running) |

## v1 results

Run on Kaggle, Tesla T4, FP16, SDPA,
transformers 4.57.1, huggingface_hub 0.36.2, torch 2.11.0, model revision `8faed761`, seed 6304. Total run time 2.7 h.
**64/64 built-in intended-vs-actual checks passed** ([`results_v1/checks.json`](results_v1/checks.json)).

Data: the official `atml-pa4-datasets` manifests were not available, so the notebook builds equivalent fixed selections
with seed 6304 and saves every ID and token window in [`results_v1/manifests/manifests.json`](results_v1/manifests/manifests.json):
128×512 C4-train calibration windows, 32 disjoint dev windows, 50,000 scored tokens each of WikiText-2 test and C4
validation, and 2,000 HellaSwag validation rows (IDs are row positions, because the `ind` field repeats).

## Main results

| Condition | Eligible sparsity / bits | Eligible MB (packed or sparse-16) | WikiText-2 PPL | C4 PPL | HellaSwag acc_norm (n) |
|---|---|---|---|---|---|
| FP16 baseline | 0 / 16 | 2499 | 11.94 | 16.49 | 0.684 (2000) |
| P1 magnitude 25% | 0.25 | 2030 | 14.90 | 19.77 | 0.697 (512) |
| P1 magnitude 50% | 0.50 | 1406 | 95.67 | 110.24 | 0.406 (2000) |
| P1 magnitude 75% | 0.75 | 781 | 1.5e6 | 2.0e6 | 0.272 (512) |
| P2 Wanda 25% | 0.25 | 2030 | 12.35 | 16.98 | 0.690 (512) |
| P2 Wanda 50% | 0.50 | 1406 | 18.69 | 23.61 | 0.579 (2000) |
| P2 Wanda 75% | 0.75 | 781 | 1685 | 1442 | 0.281 (512) |
| P3 non-uniform Wanda 50% (OWL, λ=0.04) | 0.50 | 1406 | 18.52 | 23.14 | 0.577 (2000) |
| P3 non-uniform Wanda 75% (OWL, λ=0.08) | 0.75 | 781 | 1010 | 605 | 0.289 (512) |
| Wanda 2:4 | 0.50 | 1406 | 50.24 | 60.63 | 0.438 (512) |
| Q1 RTN INT8 | 8 | 1279 | 11.93 | 16.50 | 0.688 (512) |
| Q1 RTN INT4 | 4 | 654 | 13.47 | 18.39 | 0.671 (2000) |
| Q1 RTN INT3 | 3 | 498 | 28.10 | 33.28 | 0.563 (2000) |
| Q2 AWQ-style INT8 | 8 | 1279 | 11.94 | 16.50 | 0.688 (512) |
| Q2 AWQ-style INT4 | 4 | 654 | 12.87 | 17.54 | 0.678 (2000) |
| Q2 AWQ-style INT3 | 3 | 498 | 18.08 | 23.29 | 0.608 (2000) |
| Q3 mixed, INT4 budget | all blocks 4-bit | 654 | 12.87 | 17.54 | 0.678 (2000) |
| Q3 mixed, INT3 budget | all blocks 3-bit | 498 | 18.08 | 23.29 | 0.608 (2000) |

Full table (incl. HellaSwag acc, calibration ΔNLL, dev NLL, layer error, timing): [`results_v1/summary_all_conditions.csv`](results_v1/summary_all_conditions.csv).
Packed bytes include FP16 scales + uint8 zero-points per 128-group; the sparse format is 1 mask bit per weight + 16 bits per nonzero.

## Paired 95% bootstrap intervals ([`results_v1/paired_bootstrap.csv`](results_v1/paired_bootstrap.csv))

| Comparison | ΔPPL WikiText-2 [CI] | ΔPPL C4 [CI] | Δacc_norm, pp [CI] |
|---|---|---|---|
| P3 − P2 @50% | −0.16 [−0.21, −0.11] | −0.48 [−0.61, −0.37] | −0.2 [−1.0, +0.6] |
| P3 − P2 @75% | −675 [−805, −548] | −837 [−965, −719] | +0.8 [−2.5, +3.9] |
| Wanda − magnitude @50% | −77.0 [−85.0, −70.0] | −86.6 [−101.4, −75.1] | +17.3 [+15.2, +19.4] |
| AWQ − RTN @INT4 | −0.60 [−0.70, −0.51] | −0.85 [−1.00, −0.72] | +0.7 [−0.5, +1.8] |
| AWQ − RTN @INT3 | −10.0 [−11.0, −9.2] | −10.0 [−11.5, −8.7] | +4.6 [+2.9, +6.3] |
| 2:4 − unstructured @50% | +31.6 [+28.2, +35.4] | +37.0 [+30.7, +45.4] | −15.8 [−19.9, −11.9] |
| Q3 − Q2 (INT4 and INT3) | 0 (identical allocation) | 0 | 0 |

## Figures
`results_v1/figures/`: pruning quality vs sparsity (fig1), quantization quality vs packed MB (fig2), P3 sensitivity + allocation (fig3),
Q3 sensitivity + allocation (fig4), calibration-to-held-out transfer (fig5), Wanda/magnitude mask overlap (fig6), layer-error maps (fig7).

## Known issues in this run
* **Q3 collapsed to the uniform allocation.** Under both sensitivity measures the exact DP picked all-4-bit (INT4 budget) and all-3-bit
  (INT3 budget), so Q3 equals Q2. With block granularity and bits {2,3,4,8}, raising one block to 8 bits costs four blocks dropping 4→3,
  and the summed probe ΔNLL of that trade is always worse. This is a property of the allocation granularity, not a crash.
* **Peak-memory column for quantized conditions is inflated by ~1.2 GB.** The AWQ fold-check cell leaves `ref_logits`
  (4×512×151,936 FP32 ≈ 1,187 MiB) on the GPU; every Q condition is measured after it. Pruned and quantized models run the same dense
  FP16 kernels, so their true peak memory should equal the baseline's. Fix: `del ref_logits` at the end of that cell.
* Latency: all conditions use dense FP16 kernels (prefill-2048 ≈ 850–930 ms, decode ≈ 37 ms/token). Storage savings are packed-format
  estimates, not measured speedups.

Raw per-example scores are in `results_v1/per_example/`, per-condition records in `results_v1/results/`, the stdout of the run in
[`results_v1/run_log.txt`](results_v1/run_log.txt), and the AWQ scales/alphas in `results_v1/awq_search.pt`.
