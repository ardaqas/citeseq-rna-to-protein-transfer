# Predicting surface protein from RNA in single cells (CITE-seq)

A learning project that re-examines a known problem: how well can surface
protein abundance be predicted from RNA in single cells, and how much of that
success comes from cell-type identity versus within-cell-type variation?
It does not propose a new method.

> Status: Phases 0-2 done. Results and figures are added phase by phase.

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
  unambiguous, so their signal may be close to background (hypothesis, compared with the
  isotype-control noise reference in `results/phase2_isotype_noise_reference.csv`).
- Full table: `results/phase2_protein_rna_correlation.csv`.

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
