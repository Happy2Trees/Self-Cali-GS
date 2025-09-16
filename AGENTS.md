# Repository Guidelines

## Project Structure & Module Organization
Core training entry point `train.py` orchestrates dataset loading from `scene/` and rendering logic in `gaussian_renderer/`. Parameter schemas in `arguments/`, while shared math, camera, and logging helpers live under `utils/`. CUDA/CPP extensions reside in `3dgs-pose/` and `simple-knn/`; install them in editable mode after any change. Experiment recipes and job scripts live in `script/`, `script_cvpr/`, and `training_script/`; keep new configs alongside related assets.

## Build, Test, and Development Commands
Set up the conda environment in line with the README, then install Python deps: `pip install -r requirements.txt`. Compile native modules after edits using `pip install .` from both `3dgs-pose` and `simple-knn`. Use visdom when monitoring training: `visdom -port 8600`. Typical optimization run: `python train.py -s data/scene -m output/scene_run --opt_cam --eval`. Rendering-only jobs use `python render.py --model_path output/scene_run --sequence poses.json`. Keep `wandb_mode=offline` for local debugging.

## Coding Style & Naming Conventions
Target Python 3.10 with four-space indentation and max ~120-character lines to mirror existing files. Use snake_case for functions and modules (`utils/camera_utils.py`), CamelCase for classes (e.g., `GaussianModel`), and uppercase constants. Prefer explicit argument names and type hints where clarity benefits long training pipelines. When touching CUDA/C++ sources, follow the brace placement already used in `3dgs-pose/cuda_rasterizer`.

## Testing Guidelines
There is no dedicated unit-test suite; validate changes by running a short training iteration on an example dataset (see `example_datasets` link in README) and confirm PSNR/SSIM logs stay stable. Re-render a validation trajectory with `render_trajectory/run.sh` or `python render.py ... --eval` to catch regressions in camera or distortion modules. Capture any warnings or NaN reports from `train.py` and document mitigations in the PR.

## Commit & Pull Request Guidelines
Write concise, imperative commit subjects similar to `Update README.md`; include scope prefixes when touching specific modules (e.g., `scene:`). Each PR should describe the dataset or script impacted, GPU/driver assumptions, and before/after metrics or screenshots from `figs/`. Link related issues and mention if submodules require syncing or re-installation (`pip install .`). Request CUDA rebuild confirmation from reviewers when changing kernels.

## Data & Configuration Tips
Camera data must follow the `<scene>/{images,sparse}` structure documented in the README. Store large assets outside the repo and reference them via download links. Keep default seeds (set in `train.py`) unless there is a reproducibility reason; if modified, expose a CLI flag and document expected variance.
