# Repository Guidelines

## Project Structure & Module Organization
This repo is notebook-first. Learning content lives under `notebooks/`, either as numbered modules like `notebooks/module_00_environment_basics.ipynb` or as topic folders such as `notebooks/foundations/` and `notebooks/supervised_learning/`. Project docs are in `README.md` (setup/overview) and `intro.txt` (long-form notes).

## Build, Test, and Development Commands
- `python -m venv venv` and `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS) create and activate a virtual environment.
- `pip install torch torchvision torchaudio numpy pandas matplotlib scikit-learn jupyter notebook` installs the core runtime dependencies used across notebooks.
- `jupyter notebook notebooks/` starts the Jupyter server; open a specific file such as `notebooks/foundations/environment_setup.ipynb`.
- There is no separate build step; run notebooks directly in Jupyter.

## Coding Style & Naming Conventions
- Use 4-space indentation and PEP 8 style in Python cells; keep cells focused and avoid hidden state.
- Use `snake_case` for variables/functions and `CamelCase` for classes; name notebooks `snake_case.ipynb`.
- For new linear modules, follow the `module_XX_topic.ipynb` pattern and add short markdown headings per section.

## Testing Guidelines
- No automated test framework or coverage targets are defined in this repo.
- Validate changes by running the edited notebook end-to-end (Kernel: Restart & Run All) and checking outputs/plots.
- If you add tests in the future, document how to run them in the PR.

## Commit & Pull Request Guidelines
- Git history is minimal (only "first commit"), so no formal commit-message convention exists yet.
- Keep commit subjects short and imperative, and mention the notebook/topic changed (example: `add transformer attention demo`).
- PRs should summarize notebook changes, list new dependencies, and note runtime/GPU expectations; include screenshots when visual outputs change.

## Environment & Configuration Tips
- The notebooks expect Python 3.8+ and PyTorch 2.0+ (see `README.md`). Prefer relative paths to keep runs reproducible.
- GPU is optional, but if CUDA or large VRAM is required, note that in the notebook and PR description.
