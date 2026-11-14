# Predicting surface protein from RNA in single cells (CITE-seq)

A learning project that re-examines a known problem: how well can surface
protein abundance be predicted from RNA in single cells, and how much of that
success comes from cell-type identity versus within-cell-type variation?
It does not propose a new method.

> Status: Phase 0 (setup). Results and figures will be added phase by phase.

## Data

NeurIPS 2021 Open Problems CITE-seq (bone marrow mononuclear cells),
GEO accession GSE194122, file
`GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz`.
The data is not stored in this repository.

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
