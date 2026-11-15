# CLAUDE.md

Learning project: predicting single-cell surface protein (ADT) from RNA (GEX)
in CITE-seq, with a focus on within-cell-type variation and donor-based splits.
It re-examines a known problem and does not propose a new method.

## Working style
- Talk to the user in Turkish. Code, comments, README and commit messages in English.
- Each phase: explain (biology, math, literature) -> code in small steps ->
  interpret the output together -> 2-3 concept questions. Do not start the next
  phase without the user's approval.
- Never invent results or references. Mark guesses as verified / not verified.
- Ask before long or memory-heavy operations. Never densify the full matrix.

## Environment
- Intel Mac (x86_64), 16 GB RAM. Everything runs locally; Google Colab is only a fallback.
- `.venv` (Python 3.10). Install with
  `uv pip install --python .venv/bin/python --only-binary llvmlite --only-binary numba -r requirements.txt`.
- `numpy<2` is pinned because Phase 4 uses `torch==2.2.2` (the last Intel-Mac build), installed separately.
- Paths come from `src/citeseq/paths.py`. Reusable code goes in `src/citeseq/`, one notebook per phase in `notebooks/`.
- Fixed random seeds. Learned preprocessing (scaling, PCA, HVG selection) is fit on the training split only.
  The test split is used once, at the end.

## Data facts (verified from the file)
- `data/raw/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad` (GEO GSE194122).
  The file uses the legacy anndata categorical encoding (`obs/__categories`).
- 90,261 cells x 14,087 features: 13,953 GEX + 134 ADT, separated by `var['feature_types']`.
- **9 donors** (`obs['DonorNumber']`), **4 sites** (`obs['Site']`), **12 samples** = site x donor (`obs['batch']`).
- donor1 was measured at all 4 sites and contributes ~34% of cells. Donors 2-9 each come
  from a single site, so the donor and site effects are confounded.
- **Donor splits group by `DonorNumber`, never by `batch`.**
- `layers['counts']` holds raw counts for both blocks.
  `X` (GEX) = `counts / GEX_size_factors`, with no log. Some size factors are implausibly
  small, so normalize GEX from counts ourselves.
  `X` (ADT) = per-cell CLR over 140 features (134 + 6 isotypes).
- `obsm['ADT_isotype_controls']` is CLR-transformed, not raw.
- **Do not use `obs['GEX_phase']`** (~65% G2M, not plausible).
- `obs['is_train']` is the competition's own split. We build our own splits.
- Provided GEX QC columns were most likely computed before gene filtering (ours: slightly
  fewer genes, slightly higher % mito; correlation 0.9998 or higher).
- ADT depth has a strong site effect (verified within donor1): median ADT per cell is
  6,073 (s1), 627 (s2), 1,623 (s3) and 4,762 (s4).
- Isotype-control share of ADT counts varies ~5x by site (~0.2% site1, ~1% site3) in every
  lineage, so the background level depends on the site.
- Gene set: PF4, PPBP, ITGA2B, ITGB3, GP9, XIST and RPS4Y1 are missing while GP1BB and TUBB1 are present.
  This looks like an exclusion list rather than a count filter (hypothesis, not verified). Do not
  write "low expression" as the reason. Do not download the Multiome file.
- Only 12/45 cell types have >= 20 cells in every donor. cDC1 and T prog cycling occur only
  in donor7, and CD8+ T naive CD127+ CD26- CD101- only in donor2.

## Phase status
- Phase 0, 1, 2 done. Phase 2: `notebooks/02_protein_rna_correlation.ipynb` (peak 2.9 GB, <1 min); map frozen at 8ee25f5 (110 primary, 7 secondary, 17 unavailable).
- Phase 1 normalization: RNA = CP10k + log1p from counts; ADT target = provided per-cell CLR.
- `pynndescent<0.6` is pinned (0.7 breaks scanpy 1.11.5 neighbors on large data).
- Phase 3 done, **validation only**: `notebooks/03_baselines.ipynb`, split `results/splits/random_split.csv.gz`,
  frozen configs `results/phase3_frozen_configs.json` (ridge PCA: full rank 2000 PCs, alpha 1000; ridge HVG: alpha 1e4).
- **Test set rule:** do not evaluate test cells until all models (incl. Phase 4) are frozen; then use it once.
- Within-type metric threshold: >= 50 cells of a type in the evaluated part, for every split
  (random split: 34 types in val, 34 in test).
- Grid rule: if the best setting is on a grid edge, extend the grid using validation only and log it.
  Exception agreed with the user: MLP first hidden layer capped at 2,048 units (logged).
- Phase 3 follow-up (03b): ridge within type 0.34 -> within type x sample 0.10 (same cells 0.33 -> 0.10;
  donor1 only, within type x site: 0.37 -> 0.10). Write "sample-signature learning" only as a hypothesis.
- **Pre-registered Phase 5 expectation** is in the README (recorded before the donor split). Do not edit it
  after seeing donor-split results; add any interpretation below it instead.
  A dated correction addendum (2026-11-16) replaces it. The donor-split "within type" metric equals
  within type x sample for donors 2-9, so a value of 0.10 is mechanical. The main test is now the
  standardized bias per (cell type x test sample) group, plus "shift captured" relative to the
  cell-type mean.
- Phase 5 splits: R (random test, once), D (leave-one-donor-out, 9 folds), C (leave-one-site-out,
  4 folds; report donor1 vs other donors). Frozen hyperparameters, refit per fold on training cells only.
  A donor split never removes a site (donor1 is at all sites).

## Metrics
Per protein: Pearson over all test cells; within-cell-type Pearson (types with
enough cells, weighted and unweighted by cell count); RMSE. The cell-type-mean
baseline uses labels, and this must be stated in the report.
