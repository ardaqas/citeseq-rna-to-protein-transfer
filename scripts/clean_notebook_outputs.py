"""Remove machine-specific absolute paths from saved notebook outputs.

Replaces the repository path with a relative path and the home directory with
"~" in text outputs (stdout/stderr streams, text/plain results, tracebacks).
Numbers, figures and code cells are not touched.

Usage (from the repository root):  .venv/bin/python scripts/clean_notebook_outputs.py
"""
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
HOME = str(Path.home())
REPLACEMENTS = [(str(ROOT) + "/", ""), (str(ROOT), "."), (HOME, "~")]


def clean(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main():
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        changed = 0
        for cell in nb.cells:
            for out in cell.get("outputs", []):
                if "text" in out:
                    new = clean(out["text"])
                    changed += new != out["text"]
                    out["text"] = new
                if "traceback" in out:
                    new = [clean(t) for t in out["traceback"]]
                    changed += new != out["traceback"]
                    out["traceback"] = new
                for key in ("text/plain", "text/html", "text/markdown"):
                    if key in out.get("data", {}):
                        new = clean(out["data"][key])
                        changed += new != out["data"][key]
                        out["data"][key] = new
        if changed:
            nbformat.write(nb, path)
        print(f"{path.name:<40} outputs cleaned: {changed}")


if __name__ == "__main__":
    main()
