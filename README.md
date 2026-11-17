# Interpreting Timestamp Permutations in Dynamic Link Prediction

WSDM 2027 short-paper reproduction repository. Manuscript: **"Interpreting
Timestamp Permutations in Dynamic Link Prediction"** (anonymous submission).

This repository is self-contained for numerical reproduction: every number in
the manuscript can be recomputed from the checked-in prediction panels and
logistic fits with CPU-only scripts (`numpy` + `scipy`). The training engine
for the crossed E1/E2 diagnostic is included as well, so the models can also
be retrained from the frozen MOOC data on a CUDA GPU.

## Abstract

Timestamp permutations can change both model training and the history supplied
at evaluation. Their effects are therefore ambiguous unless these interventions
are separated and their reach into model inputs is measured. We cross training
chronology with evaluation history to obtain four conditions, A–D, and measure
input exposure and within-query discrimination. On MOOC, TCL trained with
binary cross-entropy gains 8.03 percentage points in pairwise accuracy from
original chronology under common evaluation history, whereas changing history
at fixed weights gives a 0.03-point contrast. Suppressing explicit time
encodings during both training and inference reduces the training contrast to
3.64 points; the paired reduction is 4.39 points [3.68, 5.09] across six seeds.
A two-seed DyGFormer comparison on the same queries yields a 22.34-point
training contrast and 0.38-point history contrast, with substantially greater
input exposure. Matched logistic controls further establish
chronology-dependent recency utility under historical and random negative
sampling on MOOC. These results distinguish training-procedure sensitivity,
explicit-feature dependence, and response to altered evaluation inputs,
providing a practical protocol for interpreting permutation experiments.

## Quick start (CPU-only numerical reproduction)

```bash
pip install -r requirements.txt
python code/build_extension_zip.py            # rebuilds E1_E2_extension.zip
python code/reproduce_extension.py --archive E1_E2_extension.zip
cd logistic && python reproduce_extension.py  # E3 logistic replay
```

The first script verifies all 224 E1/E2 prediction panels (192 TCL-BCE + 32
DyGFormer), reproduces every A–D cell, seed interval, exposure count, and the
two-seed DyGFormer summaries, and prints:

```
{"status": "pass", "prediction_panels": 224, "E1_models": 72, "E2_models": 12, ...}
```

The second script (in `logistic/`) verifies all 160 random/historical logistic
models, reconstructs causal TRAIN features and standardization, reproduces the
MOOC/Wikipedia recency contrasts and CSVs, and prints `PORTABLE_EXACT_REPRODUCTION_PASS`.

Both scripts are offline, use one CPU thread, and need only `numpy`/`scipy`.

## Repository layout

```
code/          engine/          E1/E2 training+inference engine (TCL, DyGFormer,
               │                CAWN, GraphMixer, DyRep; exposure tracing; te-off)
               │                reproduce_extension.py   E1/E2 numerical replay
               │                build_extension_zip.py   packages predictions/ → zip
               │                audit_e2_two_seeds.py    read-only Tesla-block audit
data/mooc/     Frozen MOOC benchmark: events, node/edge features, held-out node
               rule, val/test negatives, five timestamp permutations (order_perm*.npy)
predictions/   E1/  192 TCL-BCE test-history prediction panels (ON/OFF time encoding)
               E2/  32 DyGFormer test-history prediction panels
               E1/report.json, E2/report.json  (manuscript tables, machine-readable)
logistic/      E3 two-dataset logistic-control package: 160 fits, TRAIN/test
               features, common-support masks, RESULT.json, plan.json
evidence/      Audited JSON artifacts backing every reported number:
               E1_FULL_REPORT.json, E1_full_AUDIT.json, E1_full_INDEPENDENT_REPLAY.json,
               E1_full_inventory.json, E2_SCOPE.json, E2_TWO_SEED_REPORT.json,
               ARCHIVE_REVIEW.json, archive_accepted_pairs.json, archive_normalized_source_records.json
```

## Retraining the E1/E2 models (GPU)

The engine in `code/engine/` reproduces the crossed diagnostic from scratch on
a CUDA GPU:

- `code/engine/config.json` — frozen hyperparameters (batch size 200, Adam
  lr=1e-4, patience 10, 20–100 epochs, earliest strict maximum of validation
  AP, BCEWithLogitsLoss).
- `code/engine/units.json` — the 36 unit definitions (6 seeds × 6
  chronologies) used by the two-seed DyGFormer block.
- `code/engine/bin/full_engine.py` — restartable training stages;
  `full_backend.py` — the DyGFormer adapter with input-exposure tracing;
  `backend.py` — the generic model adapter (TCL/DyGFormer/CAWN/GraphMixer/
  DyRep); `exposure.py` — label-free input tracing; `data_runtime.py` —
  frozen full-stream loading and the deterministic SplitMix64 negative
  schedule.
- `code/engine/source/models/` — model implementations
  (TCL.py, DyGFormer.py, CAWN.py, GraphMixer.py, MemoryModel.py, modules.py).

Recorded environments: `code/engine/environment_tesla_v1.json` (torch
2.13.0+cu130, numpy 2.2.6, sklearn 1.8.0, scipy 1.17.1, python 3.12.13) and
`environment_h200_v1.json`. The te-off ablation zeroes the time encoder's
output during training and inference while keeping the module and parameter
layout in place (see the paper, §"Explicit time encoding modifies the training
contrast").

The E1 models (72) and E2 models (12) themselves are not checked in; the 224
prediction panels produced by the frozen evaluation protocol are checked in
under `predictions/` and are sufficient for the numerical replay above.

## Data

- **MOOC** (Kumar et al., KDD 2019, JODIE) interaction benchmark,
  https://snap.stanford.edu/jodie/. `data/mooc/` contains the frozen split
  contract (`contract.json`), events, node/edge features, val/test negatives,
  and the five timestamp permutations used in the paper. Retain source terms
  and pseudonymous identifiers.
- **Wikipedia** (JODIE) is used by the logistic controls; its events and
  features are inside `logistic/reference/`.

## License

MIT — see [LICENSE](LICENSE).