# Interpreting Timestamp Permutations in Dynamic Link Prediction

WSDM 2027 short-paper reproduction package. The manuscript and all supporting
artifacts for the paper **"Interpreting Timestamp Permutations in Dynamic Link
Prediction"**.

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

## Repository layout

```
paper/    LaTeX sources for the manuscript (paper.tex, tables, figure, bib)
          plus the compiled paper.pdf (5 pages: 4 main + ethics/references).
code/     Reproduction scripts for the E1/E2 prediction extension.
evidence/ Audited JSON artifacts backing every reported number.
```

### paper/

- `paper.tex` — main manuscript; inputs `numbers.tex`, `crossed_table.tex`,
  `sampling_table.tex`, `diagnostic.pdf`, and `references.bib`.
- `paper.pdf` — compiled camera-ready-style PDF.
- `acmart.cls`, `ACM-Reference-Format.bst` — ACM template files required to
  rebuild the PDF.

To rebuild the PDF:

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### code/

- `reproduce_extension.py` — numerical replay of the E1/E2 prediction
  extension. No training or model loading; depends only on `numpy`. Verifies
  every reported panel metric, seed interval, and two-seed DyGFormer summary
  against the packaged zip:

  ```bash
  python reproduce_extension.py --archive E1_E2_extension.zip
  ```

  The extension zip is not included in this repository; it contains the 224
  prediction panels (72 TCL-BCE models, 12 DyGFormer models) referenced by the
  paper. Contact the authors for the full artifact.

- `audit_e2_two_seeds.py` — read-only replay of the two complete Tesla seed
  blocks (seeds 1004, 1005) against their source tree on `/mnt/data`; never
  dispatches jobs. Used during the internal audit of the DyGFormer transfer.

### evidence/

Independent, frozen audit artifacts (JSON) used to check every number in the
manuscript:

| File | Contents |
|---|---|
| `E1_FULL_REPORT.json` | Full E1 TCL-BCE ON/OFF report: 72 selected models, 192 unique test panels, seed-level contrasts and intervals. |
| `E1_full_AUDIT.json` | Independent audit verdict for the E1 report (replayed panels, tolerances, decision). |
| `E1_full_INDEPENDENT_REPLAY.json` | Second independent replay of E1 panels with numerical differences. |
| `E1_full_inventory.json` | Inventory of E1 run units, package manifest hashes, test panels. |
| `E2_SCOPE.json` | Scope definition for the two-seed DyGFormer block (seeds 1004/1005, five permutations, 12 models, 32 panels). |
| `E2_TWO_SEED_REPORT.json` | Full two-seed DyGFormer report with summaries, rows, and exposure audit. |
| `ARCHIVE_REVIEW.json` | Reconstruction and selection review of the 195-pair DyGLib archive (TCL/DyGFormer/CAWN/DyRep/GraphMixer on Wikipedia/Reddit/MOOC/LastFM). |
| `archive_accepted_pairs.json` | The 195 accepted archive pairs (model, dataset, seed, native/shuffle AP). |
| `archive_normalized_source_records.json` | 645 normalized source records used to build the archive table. |

## Data and models

- **MOOC** (Kumar et al., KDD 2019) and **Wikipedia** (JODIE) interaction
  benchmarks; see `references.bib`.
- TCL (Wang et al., 2021) in a DyGLib-based implementation (Yu et al., 2023),
  trained with binary cross-entropy ("TCL-BCE").
- Full-batch L2-regularized logistic controls with counts/seen and
  recency (`log(1+age/τ)`) feature sets.

The heavy training artifacts (model checkpoints, 224 prediction panels) are not
distributed in this repository; the `code/` scripts replay and verify them when
provided with the artifact zip.

## License

MIT — see [LICENSE](LICENSE).