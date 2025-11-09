# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `metdata/`: `gwo.py` handles Ground Weather Observation parsing, `whp.py` exposes Weather-Hub utilities, and `gwo_stn.csv` stores station metadata consumed by both modules. Notebooks and runnable samples (e.g., `gwo_hourly_example.ipynb`, `export_gwo_gotm.sample.py`) sit in `examples/` for exploratory work. Packaging files (`setup.py`, `pyproject.toml`, `setup.cfg`) plus environment specs (`requirements.txt`, `environment.yml`) remain at the repo root so tooling can locate them easily.

## Build, Test, and Development Commands
Use conda when possible:
```bash
conda env create -f environment.yml
conda activate metdata
pip install -e .
```
Pip-only workflow:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
Both flows require the system package `nkf`; install it via `sudo apt install nkf` (Linux) or `brew install nkf` (macOS). During development, point `DATA_DIR` to your external GWO data path so `metdata.gwo` locates hourly CSVs without hard-coded paths.

## Coding Style & Naming Conventions
Follow PEP8 with 4-space indentation, `snake_case` functions, and `CapWords` classes (e.g., `Data1D_PlotConfig`). Keep modules single-purpose to match `gwo.py`’s current layout, and colocate constants (like `rmk`) near their consumers. Prefer pandas/numpy vectorized logic and guard platform-specific behavior (e.g., `_get_default_gwo_path`) behind helper functions. Include docstrings describing inputs, JMA terminology, and expected units.

## Testing Guidelines
There is no automated suite yet; add `tests/` alongside `metdata/` and name files `test_*.py`. Use `pytest` as the runner:
```bash
pytest -q
```
Focus on deterministic components (parsers, path helpers, rolling-window math) and keep fixture CSVs lightweight. Document any reliance on external data directories so tests degrade gracefully when `DATA_DIR` is absent.

## Commit & Pull Request Guidelines
Existing history favors short, imperative commits (e.g., "Updated reflecting GWO-AMD"). Keep that tone and scope: describe *what* changed and *why* in under ~72 characters, elaborating in the body only when necessary. Pull requests should summarize the feature, list affected commands or data paths, link related issues, and attach plots/logs when behavior is data-dependent. Call out validation steps (commands run, sample stations processed) so reviewers can reproduce results quickly.

## Data & Configuration Tips
Store raw GWO/AMD CSVs outside the repo and pass the absolute path through `DATA_DIR` or explicit constructor arguments. Never commit licensed datasets. Verify locale-dependent tools (`nkf`, matplotlib fonts) locally before opening a PR, noting any OS-specific tweaks in the description.
