# Analysis log (phase by phase)

> This is the full, chronological project log. It was moved here unchanged from the README when the README was rewritten in Phase 6. It includes the pre-registered Phase 5 expectations, committed in `fa39914` before any donor-split or site-split result.


A learning project that re-examines a known problem: how well can surface
protein abundance be predicted from RNA in single cells, and how much of that
success comes from cell-type identity versus within-cell-type variation?
It does not propose a new method.

> Status: Phases 0-5 done (Phase 5 = final test evaluation, run once). Results and figures are added phase by phase.

## Data

NeurIPS 2021 Open Problems CITE-seq (bone marrow mononuclear cells),
GEO accession GSE194122, file
`GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz`.
The data is not stored in this repository.

Verified from the file (see `results/data_structure.txt`):

- 90,261 cells; 14,087 features = 13,953 genes (`var['feature_types'] == 'GEX'`)
  + 134 surface proteins (`'ADT'`) in one matrix.
- **9 donors** (`obs['DonorNumber']`), **4 sites** (`obs['Site']`),
  **12 samples** = site x donor (`obs['batch']`, e.g. `s3d7`).
- 45 annotated cell types (`obs['cell_type']`), from 21,693 cells (CD14+ Mono)
  down to 18 (cDC1).

### Donor and site design

| Site | Donors | Cells |
|---|---|---|
| site1 | donor1, donor2, donor3 | 16,311 |
| site2 | donor1, donor4, donor5 | 25,171 |
| site3 | donor1, donor6, donor7 | 32,029 |
| site4 | donor1, donor8, donor9 | 16,750 |

Cells per donor and site (`crosstab(DonorNumber, Site)`):

| donor | site1 | site2 | site3 | site4 | total |
|---|---|---|---|---|---|
| donor1 | 5,227 | 10,465 | 9,521 | 5,456 | 30,669 |
| donor2 | 4,978 | 0 | 0 | 0 | 4,978 |
| donor3 | 6,106 | 0 | 0 | 0 | 6,106 |
| donor4 | 0 | 5,584 | 0 | 0 | 5,584 |
| donor5 | 0 | 9,122 | 0 | 0 | 9,122 |
| donor6 | 0 | 0 | 11,035 | 0 | 11,035 |
| donor7 | 0 | 0 | 11,473 | 0 | 11,473 |
| donor8 | 0 | 0 | 0 | 3,929 | 3,929 |
| donor9 | 0 | 0 | 0 | 7,365 | 7,365 |
| total | 16,311 | 25,171 | 32,029 | 16,750 | 90,261 |

- donor1 was measured at all four sites and contributes 30,669 cells (~34%).
- Because donor1 is at every site, holding out any single donor never removes a site from training.
- Every other donor was measured at a single site, so for donors 2-9 the donor
  effect and the site effect cannot be separated (donor-site confounding).
  A held-out donor is therefore also, mostly, a held-out site.
- Donor-based splits group by `DonorNumber`, never by `batch`: splitting by
  `batch` would put donor1 cells in both train and test.

### What `X` and `layers['counts']` contain

Checked in `notebooks/00_normalization_check.ipynb` by reconstructing `X` from
`counts`:

| Block | `layers['counts']` | `X` |
|---|---|---|
| GEX | raw UMI counts | `counts / obs['GEX_size_factors']` (no log) |
| ADT | raw ADT counts | per-cell CLR, `log1p(x / exp(mean_j log1p(x_j)))` over 140 features (134 proteins + 6 isotype controls) |

The original competition documentation describes the same two steps (scran
size factors for GEX, CLR for ADT), but its QC thresholds do not match this file,
so the file was checked directly. Some GEX size factors are implausibly small
(minimum ~0.0002), so this project normalizes GEX from `layers['counts']` itself.

### Site effects on protein measurements

- ADT depth per cell differs strongly by site. Within donor1 (the only donor measured at all
  sites) the median is 6,073 at site1 vs 627 at site2.
- The isotype-control share of ADT counts varies about 5-fold by site (~0.2% at site1, ~1% at
  site3) in every lineage. The **background level depends on the site**.

### Protein-gene mapping (frozen before results)

`resources/protein_gene_map.csv` maps each of the 134 proteins to its gene(s) by hand.
The mapping and the split into analysis sets were committed **before** any
full-data correlation was computed.

- When a protein maps to several genes, the raw counts of those genes are summed
  before normalization (HLA-A/B/C, TRAC/TRBC1/TRBC2, FCGR2A/FCGR2B).
- **Primary set (110 proteins):** direct mappings plus unambiguous complexes
  (CD3 -> CD3E, CD8 -> CD8A, HLA-A-B-C, HLA-DR -> HLA-DRA, TCR). Headline results use only this set.
- **Secondary set (7 proteins), reported separately:** CD45, CD45RA, CD45RO (all
  compared with the same PTPRC RNA, differing only by splicing), CD107a, CD63
  (mostly intracellular), CD32, CD16 (paralog ambiguity).
- **Unavailable (17 proteins):** at least one mapped gene is not among the 13,953
  genes in the file. Why these genes are missing has **not** been determined
  (see below). Results for 110 or 117 proteins do not generalize to all 134: the missing
  proteins may be a systematically harder subset. In Phase 3 (validation), ridge predicts these 17
  proteins less well than the primary set: median Pearson 0.47 vs 0.73 over all cells, and 0.25 vs
  0.34 within cell type. The label-based cell-type mean is also lower for them (0.39 vs 0.65).

**Why are some genes missing? (not resolved)**

- The organizers' per-sample analysis notebooks
  ([openproblems-bio/neurips2021-notebooks](https://github.com/openproblems-bio/neurips2021-notebooks))
  filter genes with `sc.pp.filter_genes(min_cells=20)` within each sample.
- The final gene set is **not** explained by "at least 20 cells in every sample":
  1,749 retained genes fall below that in at least one sample.
- No obvious single global threshold either. The least-detected retained gene is found
  in 172 cells.
- A simple low-expression filter also cannot explain the missing genes. Strongly expressed
  platelet/megakaryocyte genes (PF4, PPBP, ITGA2B, ITGB3, GP9) and sex-linked genes
  (XIST, RPS4Y1) are absent, while related genes such as GP1BB and TUBB1 are present.
- **Hypothesis (not verified):** this pattern looks more like an exclusion list (specific genes
  removed on purpose) than a count-based filter.
- Unexplained side observation: several genes near the start of chromosome 1 are detected in 0 cells of
  sample `s3d1` only.

Mapping changes:

| Date | Change | Timing |
|---|---|---|
| 2026-11-15 | CD32: FCGR2A -> FCGR2A + FCGR2B | before results (subset results were not inspected) |

Any change made after seeing results will be marked **post hoc** here.

## Results so far

### Phase 2: protein vs. its own gene's RNA

`notebooks/02_protein_rna_correlation.ipynb`, primary set of 110 proteins. Values are medians
across proteins (Pearson; RNA = log1p CP10k, protein = per-cell CLR).

| Correlation | median (IQR) |
|---|---|
| all cells | 0.29 (0.14-0.45) |
| between cell types (pseudo-bulk) | 0.84 (0.61-0.94) |
| within cell type, cell-weighted | 0.07 (0.04-0.12) |
| within cell type x sample | 0.06 |

- Most of the own-gene correlation comes from differences between cell types.
  Within a cell type it is small for nearly all proteins (lower than overall for 100 of 110).
- Spearman gives the same picture, and controlling for sample barely changes the within-type values.
- A mapping diagnostic flags 8 proteins whose own-gene correlation is ~0 and ranks below
  4,000 of 13,953 genes (CD83, CD115, CD103, CD163, CD154, CD223, CD152, CD42b). Their mappings are
  unambiguous. Follow-up (`notebooks/02b_phase2_followups.ipynb`):
  - Only CD103 and CD42b sit at isotype-control level, which fits a background-level signal.
  - The other six are above isotype level in most cells but still do not follow their own gene's RNA.
    Why is not known.
  - The CD115 signal is higher in erythroid cells than in CD14+ monocytes, even though CSF1R is detected
    in 25% of CD14+ Mono and essentially not in erythroid cells.
- ALAS2 (an erythroid gene) was the top-correlated gene for all isotype controls and 15 proteins.
  This is an erythroid-vs-rest difference. Without erythroid cells, or within cell types, ALAS2's
  correlation is about 0. The lower isotype share in erythroid cells may relate to Fc receptors
  (hypothesis, not tested).
- Tables: `results/phase2_protein_rna_correlation.csv`, `results/phase2b_alas2.csv`,
  `results/phase2b_low_correlation_proteins.csv`.

### Phase 3: baselines (validation only)

> **These are validation numbers only. The test set has not been used.** It will be used once,
> after all models (including Phase 4) are frozen. Ridge hyperparameters were chosen on the same
> validation cells, so its numbers are slightly optimistic.

`notebooks/03_baselines.ipynb`. Random cell split 70/15/15, stratified by cell type
(`results/splits/random_split.csv.gz`). HVGs, scaling and PCA were fit on training cells only. Primary set
(110 proteins), median [IQR] across proteins. Within-type values use the 34 cell types that have
>= 50 validation cells.

| Model | Pearson, all cells | within cell type (weighted) | RMSE |
|---|---|---|---|
| global mean | undefined | undefined | 0.51 |
| cell-type mean (**uses the cell-type labels**) | 0.65 [0.47-0.73] | undefined | 0.35 |
| own gene (linear) | 0.29 [0.15-0.45] | 0.08 [0.04-0.13] | 0.45 |
| ridge on PCA (selected: full rank, 2,000 PCs) | 0.73 [0.58-0.83] | 0.34 [0.27-0.43] | 0.31 |
| ridge on z-scored HVGs | 0.73 [0.57-0.83] | 0.33 [0.26-0.42] | 0.31 |

- "Undefined" means the correlation does not exist for a constant prediction. It is not counted as 0.
- Ridge (RNA only, no labels) beats the label-based cell-type mean overall for 109 of 110 proteins.
  It beats the own-gene model on both metrics for all 110.
- The PCA grid {50, ..., 2,000} was best at full rank, which is the upper bound (logged in
  `results/phase3_frozen_configs.json`). Sanity check: ridge on full-rank PCA equals ridge on the
  unscaled HVGs (max difference 2.4e-6).
- Within-type metric threshold, fixed before any test evaluation: >= 50 cells of a type in the evaluated
  part, for every split. Random split: 34 evaluable types in validation and 34 in test. The donor split
  is reported in Phase 5.
- **Within cell type x sample** (`notebooks/03b_phase3_followups.ipynb`, validation). Holding the sample
  fixed as well lowers ridge's within-type Pearson from 0.34 to 0.10. The own gene barely changes
  (0.08 -> 0.06; Phase 2: 0.07 -> 0.06).
  - Of ridge's 0.34 within cell type, about **0.23 is at the sample level** (technical and donor
    differences, which this design cannot separate). About **0.10 is cell-to-cell variation within a
    sample**. This is a heuristic split (a difference of two correlations), not an exact variance
    decomposition.
  - The result holds on exactly the same cells (8,454 cells in 72 type x sample groups): 0.33 -> 0.10.
    The own gene stays at 0.06.
  - **donor1 only** (one donor, four sites): ridge 0.37 within type -> 0.10 within type x site. This uses
    4,635 validation cells: 20 cell types and 24 cell type x site groups with >= 50 cells. On the same
    cells (2,892 cells, 11 types, 24 groups) it is 0.34 -> 0.10. With the donor fixed, most of the
    between-sample part comes from site, i.e. it is technical, at least for donor1.
  - *Hypothesis, not established:* ridge may pick up sample/site signatures in RNA. Alternatively, site
    may affect RNA and protein together. This was not tested.
- **PCA components** (validation): with 50 PCs ridge keeps 95% of its full-rank overall Pearson
  (0.695 vs 0.733, absolute difference 0.038). It keeps only 81% of its within-type Pearson (0.274 vs
  0.337, difference 0.063) and 85% of its within type x sample Pearson (0.088 vs 0.104, difference 0.016).
  PCA truncation costs relatively more within-type information. In absolute terms the within-sample part
  is small.
- Other open hypothesis: part of the within-type signal may be finer structure inside annotated types.

### Pre-registered expectation for Phase 5 (written before any donor-split result)

Recorded 2026-11-15, before the donor split was built or evaluated.

- **Metric:** ridge's within-cell-type Pearson (median over the 110 primary proteins, cell-weighted) on
  held-out donors, compared with the random-split validation values: 0.34 within type, 0.10 within
  type x sample.
- **If it approaches 0.10**, this supports the sample-signature hypothesis: the extra within-type
  signal in the random split comes from samples seen in training.
- **If it stays near 0.34**, ridge learns something that generalizes across samples.
- Values in between are read as partial support. Within type x sample will be reported for the donor
  split too.
- Caveat known in advance: a donor split also removes those donors' cells from training and, except for
  donor1, holds out their site. A drop can therefore have more than one cause.

#### Correction addendum (2026-11-15, still before any donor-split or site-split result)

The expectation above is flawed. It is kept unchanged for the record and is replaced by the one below.

1. **The 0.10 comparison is mechanical.** donor2-9 each consist of a single sample. For a test set
   holding one of them, "within cell type" Pearson **is** "within cell type x sample" Pearson. That
   metric is 0.10 in the random split too. A donor-split value near 0.10 is therefore expected by
   construction and is **not** evidence for the sample-signature hypothesis.
2. **The caveat about sites was wrong.** donor1 was measured at every site, so a donor split never removes
   a site from training (see the donor x site table above). For donors 2-9 their site stays in training
   through donor1 and the other donor of that site.

**Phase 5 design (fixed now)**

- Models use the frozen Phase 3/4 hyperparameters, with no retuning on the new splits. Within each fold,
  every learned step is refit on that fold's training cells only: HVGs, scaling, PCA, ridge weights,
  target scaling.
- Hypothesis judgments use **ridge 3a**, which is deterministic. The cell-type-mean baseline is needed
  as a reference for metric 2. The own-gene model and the MLP are reported alongside.
- **MLP seeds:** 3 seeds on split R; 1 seed (seed 0) per fold in D and C. The Phase 4 seed-to-seed SD is
  reported next to the fold results, so it can be compared with the differences between folds.
- **MLP early stopping in D and C folds:** by an **inner validation set**: 15% of the fold's *training*
  cells, drawn at random, stratified by cell type, seed 0. The MLP trains on the remaining 85%. Test-donor
  and test-site cells never enter training, early stopping, or any scaling.
  - Chosen over a fixed epoch count because it repeats the Phase 4 procedure exactly.
  - Known cost: in these folds the MLP sees 15% fewer training cells than ridge.
  - On split R the MLP keeps the Phase 4 setup: early stopping on the random-split validation part.
- Splits:
  - **R** — the existing random split, test part, used once.
  - **D** — leave-one-donor-out, 9 folds. donor2-9 folds have 1 test sample each; the donor1 fold has
    4 test samples, all at sites seen in training.
  - **C** — leave-one-site-out, 4 folds. Each test site has 3 samples. Test cells are reported in two
    groups: **donor1** (donor seen in training, site not) and **other donors** (neither seen).
- All metrics: 110 primary proteins; groups need >= 50 test cells.
- **Cell types absent from a fold's training cells** (e.g. cDC1 and T prog cycling in the donor7 fold)
  are excluded from metric 2. The cell-type mean cannot predict them, so their bias has no reference. The
  number of excluded types and cells is reported per fold.

**Metrics, for every split**

1. **Within cell type x sample Pearson** (cell-weighted over groups, median over proteins). Compared with
   the random split (validation: 0.104).
2. **Standardized bias per (cell type x test sample) group:** `b = (mean prediction - mean truth) / SD`,
   where SD is the protein's standard deviation in the fold's training cells.
   - Summary: cell-weighted mean of |b| over all (primary protein, group) pairs, per fold and per split.
   - The same number is computed for the random split.
   - Reference: the cell-type-mean baseline predicts the training mean of the type, so its |b| measures
     how far the test sample's type mean lies from training, i.e. the sample shift.
   - **Shift captured** = 1 - Σ|b|(ridge) / Σ|b|(cell-type mean). Both sums run over the same (primary
     protein, group) pairs, each |b| weighted by the group's cell count. This is a **ratio of sums**,
     not a median of per-group ratios.
3. **Pooled within cell type Pearson**, only for test sets with >= 2 samples (donor1 fold: 4 samples;
   each site fold: 3), reported together with the number of samples.

**Expectations (written before any result)**

- **Metric 1:** about 0.10 in every split under both hypotheses, so this metric does not discriminate
  them. A clear drop well below 0.10 would mean that even within-sample, cell-level prediction depends on
  having seen the sample.
- **Metric 2 (the main test):**
  - The sample-signature hypothesis predicts small |b| for samples seen in training (split R) and clearly
    larger |b| for unseen samples. Shift captured would then be near 0, i.e. close to the cell-type-mean
    baseline.
  - **Only the C split can contradict the hypothesis.** In the C folds, if ridge's |b| stays small and
    shift captured stays well above 0 on an unseen site, ridge transfers sample shifts to unseen sites.
    Even then, this is not evidence for biology: it only shows that the shift can be predicted from RNA.
  - **In D, a small |b| is not informative about the hypothesis.** The test donor's site is in training,
    so a small |b| is also consistent with a learned site signature.
- **Site vs. donor** (motivated by the donor1-only validation result):
  - If the sample-level component is mainly site-driven, the D fold for donor1 (sites seen through other
    donors) should show small |b| and a pooled within-type Pearson clearly above 0.10.
  - In the C folds, donor1 cells (unseen site) should then show |b| similar to the other donors at that
    site. If it is mainly donor-driven, donor1 cells should show smaller |b| than the other donors in
    the C folds.
- **Metric 3:** if sample shifts do not transfer, pooled within-type Pearson on unseen sites (C folds)
  should be clearly below the random-split 0.34 and closer to metric 1.

Edits to this addendum, all made before any donor-split or site-split result:
- Date corrected to 2026-11-15.
- Added: MLP early stopping by inner validation, MLP seeds, exclusion of unseen cell types from metric 2.
- Shift captured redefined as a ratio of sums.
- Metric-2 expectation narrowed: only C can contradict the hypothesis; small |b| in D is not informative.

### Phase 4: MLP (validation only)

`notebooks/04_mlp.ipynb`.
- Input: z-scored 2,000 HVGs, fit on training cells. Targets: all 134 proteins, standardized per protein.
- Training: early stopping on validation loss. The configuration is chosen by mean validation Pearson.
  The best configuration is trained with 3 seeds and reported as mean ± SD.
- **Validation optimism:** the MLP's search space (architecture, dropout, weight decay, number of epochs
  through early stopping) is wider than ridge's single alpha, and validation decides all of it. Its
  validation numbers are therefore likely to be somewhat more optimistic than ridge's. Only the test
  set, used once at the end, gives a fair comparison.

Results (validation only; primary set; median over proteins, mean ± SD over 3 seeds):

| | MLP [1024, 512], dropout 0.3 | ridge 3a | MLP - ridge |
|---|---|---|---|
| Pearson, all cells | 0.794 ± 0.001 | 0.733 | +0.061 |
| within cell type (weighted) | 0.446 ± 0.000 | 0.337 | +0.109 |
| within cell type x sample | 0.114 ± 0.003 | 0.104 | +0.010 |
| RMSE | 0.289 ± 0.001 | 0.312 | -0.023 |

- **Search:** 15 configurations, one validation-based grid extension (logged in
  `results/phase4_frozen_config.json`). The width cap was not binding.
- **Ranking by within type x sample** would pick a different configuration, but only by 0.0005, which is
  well below the seed SD. In practice it is a tie.
- **Where the MLP gains:** most of its gain over ridge within cell type (+0.11) is at the sample level.
  Within cell type x sample it gains only +0.01. Whether this part transfers to unseen samples is what
  Phase 5 tests.
- **Seeds:** seed-to-seed SD is <= 0.003 for every metric.

### Phase 5: final evaluation (test sets, evaluated once)

`notebooks/05_final_evaluation.ipynb`, run once with the design pre-registered above (commit `fa39914`).
No setting or mapping was changed afterwards. Primary set (110 proteins); ridge 3a; MLP seed 0 in D/C,
3 seeds on R (SD <= 0.006).

| Report group | within type x sample r: ridge / MLP | shift captured: ridge / MLP | overall Pearson: ridge / MLP / cell-type mean |
|---|---|---|---|
| R test (samples seen) | 0.099 / 0.109 | 0.43 / 0.62 | 0.736 / 0.792 / 0.643 |
| D donors 2-9 (donor unseen, site seen) | 0.122 / 0.131 | 0.18 / 0.25 | per fold in CSV |
| D donor1 (donor unseen, sites seen) | 0.108 / 0.121 | 0.14 / 0.22 | 0.695 / 0.735 / 0.626 |
| C donor1 cells (site unseen) | 0.109 / 0.113 | 0.05 / 0.03 | see site folds |
| C other donors (site and donor unseen) | 0.119 / 0.118 | 0.07 / 0.07 | see site folds |
| C, mean of the 4 site folds | — | — | 0.648 / 0.646 / 0.641 |

**Against the pre-registered expectations**
- **Metric 1** (about 0.10 everywhere, not discriminating): **met.** 0.10-0.13 in every group.
- **Metric 2** (only C can contradict the sample-signature hypothesis): **not contradicted.** On unseen
  sites, the models capture 3-7% of the between-sample shift, close to the cell-type-mean baseline. On the
  random split they capture 43-62%.
- **Site vs. donor: mixed.**
  - In C, donor1 cells do not show smaller bias than the other donors. This is consistent with a
    site-driven shift and does not support a donor-driven one.
  - But the D-donor1 fold does not show the small bias predicted for a site-driven shift (shift
    captured 0.14). Its pooled within-type r (0.20) is above 0.10, as predicted.
- **Metric 3** (unseen sites clearly below 0.34, closer to metric 1): **met.** Ridge 0.07-0.19 per site
  fold (mean 0.13) vs 0.34 on the random split.
- **Disclosed deviation:** metric 3 for whole site folds was omitted in the first run and added afterwards
  from saved predictions, without refitting.

**What this says about the research question (interpretation)**
- With a random split, RNA predicts protein far better than cell-type identity, both overall and within
  cell types. But much of the within-type advantage comes from predicting sample-level shifts for samples
  seen in training.
- On an unseen site:
  - That shift is not transferred.
  - The multi-gene models are only marginally better than the label-based cell-type mean overall.
  - The MLP's advantage over ridge disappears.
- What does carry over is a modest within-sample, cell-level signal (within type x sample r about 0.11-0.12),
  well above the protein's own gene (about 0.06).
- Whether the non-transferable shift is technical (e.g. site-dependent ADT background) or biological
  cannot be settled with this design. The site-split results point more to site than to donor.
- Overall accuracy on unseen sites ranges from 0.45 (site3) to 0.81 (site1). The two weakest sites are the
  ones with low ADT depth and a high isotype share; that this explains it is a hypothesis.
- Tables: `results/phase5_report_groups.csv`, `results/phase5_per_fold.csv`, `results/phase5_m3_site_folds.csv`,
  `results/phase5_excluded_cell_types.csv`.

`obs['GEX_phase']` (cell-cycle phase) is not used: it labels ~65% of bone marrow
cells as G2M, which is not plausible.

## Limitations (running list)

- **Hyperparameters were frozen on the random split.** That condition rewards sample signatures, and the
  frozen settings were chosen on validation cells from every donor. They are reused unchanged in the donor
  and site splits.
- **The donor1 fold of the donor split** removes 34% of the training cells and the only donor that links
  the four sites.
- **The 17 proteins without a measured gene** look harder: Phase 3 validation, ridge 0.47 vs 0.73 overall
  and 0.25 vs 0.34 within cell type. Results on 110/117 proteins do not generalize to all 134.
- **Donor and site are confounded** for donors 2-9 (one site each). Only donor1 allows a within-donor site
  comparison.
- **Validation numbers are optimistic,** more so for the MLP than for ridge (wider search, early
  stopping on validation).
- **"Within cell type" means within the 45 provided labels.** These labels come from the data providers
  and are themselves partly based on RNA and protein.

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python \
    --only-binary llvmlite --only-binary numba -r requirements.txt
.venv/bin/python -m ipykernel install --user --name citeseq
```

Download the data (~616 MB compressed) into `data/raw/` and decompress:

```bash
mkdir -p data/raw && cd data/raw
curl -L -C - -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
gunzip GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
```

PyTorch is installed separately in Phase 4.

## Layout

```
data/raw/        downloaded data (gitignored)
notebooks/       one notebook per phase
src/citeseq/     reusable functions
results/figures/ figures
tests/           unit tests
```
