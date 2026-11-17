Anonymous E3 numerical reproducibility extension
Interpreting Timestamp Permutations in Dynamic Link Prediction

Run with Python 3.14, NumPy 2.4.2 and SciPy 1.17.1:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python reproduce_extension.py

The script verifies every packaged input hash, reconstructs TRAIN causal
features and standardization for80 random models, verifies160 TEST score
panels from coefficients, and reproduces four CSVs exactly. All outputs go
to reproduced/. No model is fitted; no network or external base artifact is
required. Linux address space is bounded at4GiB; use one CPU thread.

reference/: original benchmark events, split contract, minimal frozen
TRAIN permutation/support arrays, TEST common masks,80 historical models.
data/: random/historical TEST features and candidates; fixed TRAIN query
IDs and random candidates, for20 paired dataset/draw cells.
fits/:80 random models,12 reused from validation POC and68 newly fitted,
with TEST logits and sanitized results. historical_test/:80 TEST panels.
recorded/: independently verified reference CSVs and verification result.
RESULT.json: complete scientific results including every failed gate.
plan.json: scientific protocol with original frozen plan hash; operational
paths are removed. No author, machine or private credential is required.

The random task uses a fixed TRAIN-destination vocabulary and retains the
same restricted TRAIN common support as historical fits. TEST rows use the
historical common mask. Strategies define different tasks; compare chronology
contrasts within each strategy. All ten draws are conditional on each fixed
dataset. Validation selected the POC; these TEST sets were previously studied.
No independent-new-data confirmation is claimed. Wikipedia fails the frozen
nondegeneracy rule; its effects and all subgroup results are retained.

Source benchmarks: JODIE (Kumar et al., KDD2019),
https://snap.stanford.edu/jodie/ ; preserve pseudonymous IDs and source terms.
