# Predicting surface protein from RNA in single cells (CITE-seq)

A learning project that re-examines a known problem: how well can surface
protein abundance be predicted from RNA in single cells, and how much of that
success comes from cell-type identity versus within-cell-type variation?
It does not propose a new method.

> Status: Phases 0-3 done (Phase 3: validation only). Results and figures are added phase by phase.

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

- donor1 was measured at all four sites and contributes 30,669 cells (~34%).
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
  proteins may be a systematically harder subset.

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
- Open questions (hypotheses): part of the within-type signal may be finer structure inside annotated
  types, or sample/site signatures shared between training and validation in a random split.

`obs['GEX_phase']` (cell-cycle phase) is not used: it labels ~65% of bone marrow
cells as G2M, which is not plausible.

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
