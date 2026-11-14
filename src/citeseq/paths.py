"""Central place for file locations, so no notebook hard-codes a path.

By default everything lives inside the repo. Set the environment variable
CITESEQ_DATA_DIR to point the data folder somewhere else (e.g. Google Drive
on Colab, which is the backup plan for this project).
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CITESEQ_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# GEO GSE194122, NeurIPS 2021 Open Problems CITE-seq (BMMC), processed file.
CITE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
)
CITE_H5AD = RAW_DIR / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
