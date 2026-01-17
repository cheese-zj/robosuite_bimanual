# AGENTS.md

Guidance for agentic coding in `robosuite_bimanual`.
Focus on Python scripts and robosuite environments.

## Rule sources
- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md` found.
- Follow this file and the existing code style.

## Quick repo map
- `collect_robosuite.py`: teleop data collection.
- `collect_scripted.py`: scripted demos (incl. TwoArmClothFold).
- `train_act.py`: ACT training entry point.
- `deploy_quad.py`: bimanual deployment script (legacy name).
- `envs/`: custom two-arm robosuite environments.
- `mirror_transform.py`: mirroring utilities for augmentation.
- `augment_data.py`: mirror augmentation for datasets.
- `dataset.py`: HDF5 dataset loader.

## Environment setup
- Python 3.9+ required (see `pyproject.toml`).
- Recommended: `uv venv` then `source .venv/bin/activate`.
- Install runtime deps:
  - `uv pip install robosuite numpy h5py torch mujoco`
- Optional dev deps: `uv pip install matplotlib tensorboard tqdm`.
- SpaceMouse support: `uv pip install hidapi`.

## Build / run commands
- Build step: none (scripts only).
- View environment: `python collect_robosuite.py --demo --task TwoArmLift --robot Panda`.
- Collect teleop demos: `python collect_robosuite.py --task TwoArmLift --episodes 50`.
- Collect scripted demos: `python collect_scripted.py --task TwoArmLift --episodes 50`.
- Scripted cloth demos: `python collect_scripted.py --two_arm_cloth --policy cloth_fold --episodes 50 --render`.
- Train ACT policy: `python train_act.py --data_dir data/bimanual --epochs 500`.
- Deploy bimanual policy: `python deploy_quad.py --checkpoint checkpoints/best_model.pt --task TwoArmLift --render`.
- Mirror augmentation: `python augment_data.py --input_dir data/bimanual --output_dir data/bimanual_mirrored`.
- Offscreen rendering: `export MUJOCO_GL=egl` (or `osmesa`).

## Suggested workflow
- Collect demos into `data/bimanual`.
- Train and save models under `checkpoints/`.
- Use mirror augmentation for symmetric tasks.
- Deploy with `deploy_quad.py` (legacy name) or your own loop.

## Common runtime flags
- `--task`: robosuite task name (e.g., `TwoArmLift`).
- `--two_arm_cloth`: use the custom TwoArmClothFold environment.
- `--env_configuration`: cloth layout (`parallel`/`opposed`/`single-robot`).
- `--robot`: robot model (e.g., `Panda`).
- `--episodes`: number of episodes to collect.
- `--device`: training device override.
- `--input_ref_frame`: `base` or `world` controller frame.
- `--include_object_obs`: add cloth/object observations.
- `--render` / `--demo`: enable viewer output.

## Lint / format commands
- No formatter or linter configured in repo.
- Keep changes PEP 8-ish; do not add new tooling unless requested.
- If explicitly asked to format: `python -m black <files>`.
- If explicitly asked to lint: `python -m ruff check <files>`.

## Tests
- No project tests are present in this repo.
- If tests are added, standard pytest invocation works:
  - `python -m pytest`
  - Single test: `python -m pytest path/to/test_file.py::TestClass::test_name`
  - Pattern filter: `python -m pytest -k 'pattern'`

## Code style: general
- Use 4-space indentation and PEP 8 spacing.
- Use double-quoted docstrings for modules/classes/functions.
- Prefer short helper functions for transformations.
- Use `Path` from `pathlib` over string paths.
- Use f-strings for user-facing messages and errors.
- Keep line length reasonable (~100 chars).
- Avoid trailing whitespace and unused imports.

## Code style: imports
- Order imports: standard library, third-party, then local modules.
- Group imports with a blank line between groups.
- Keep each import on its own line when many symbols.
- Avoid wildcard imports.
- Use `from __future__ import annotations` in new env modules if matching file style.

## Code style: typing
- Use type hints for public functions and dataclasses.
- Use `Optional[T]` when `None` is allowed.
- Use `Dict[str, ...]` for observation/action mappings.
- Use `np.ndarray` for NumPy arrays and `torch.Tensor` for Torch tensors.
- Do not over-annotate internal locals.

## Code style: naming
- Classes and dataclasses: `CamelCase`.
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Keep CLI arg names consistent with existing scripts.
- Use descriptive names for transforms and configs.

## Error handling
- Validate array shapes and expected sizes early.
- Raise `ValueError` with clear messages on invalid inputs.
- Avoid silent `except` blocks.
- Use `assert` only for internal invariants.
- Prefer early returns for `None` or empty inputs.

## Numpy / Torch patterns
- Convert inputs with `np.asarray(...).copy()` before in-place changes.
- Prefer vectorized operations over Python loops when reasonable.
- Use `torch.no_grad()` for inference-only paths.
- Move tensors to device once and reuse.
- Normalize observations/actions using stored stats.

## Dataset / HDF5 conventions
- Use `h5py.File(..., 'r')` in a context manager.
- Read episode metadata from `metadata` attrs.
- Observation keys follow `robot{idx}_<field>` naming.
- Keep mirror/augmentation scripts consistent with HDF5 schema.
- Keep sample normalization in `dataset.py` consistent.

## CLI script conventions
- Use `argparse` and keep defaults aligned with README.
- Print high-level progress with `print` (no logging framework).
- Prefer explicit boolean flags like `--render` / `--two_arm_cloth`.
- Keep `if __name__ == "__main__":` entry blocks in scripts.

## Robosuite env conventions
- Use `load_composite_controller_config` for controllers.
- Support `input_ref_frame` in controller configs.
- For custom two-arm envs, use `TwoArmEnv` and `env_configuration` values.
- Keep env config in dataclasses where possible.
- Use `deepcopy` when mutating controller configs.

## File/dir hygiene
- Do not commit large data in `data/` or models in `checkpoints/`.
- Avoid editing `.venv/` or generated artifacts.
- Use `data/` for demo output and `checkpoints/` for models.
- Keep datasets in HDF5 format consistent with README.

## Documentation
- Update `README.md` when CLI flags or workflows change.
- Add inline comments only for non-obvious math or transforms.

## When unsure
- Check `README.md` for usage examples and CLI flags.
- Mirror transformations are in `mirror_transform.py`.
- Two-arm cloth env logic lives in `envs/two_arm_cloth_fold.py`.
- Ask before introducing new dependencies or tools.

## Notes for agents
- This repo is script-centric; keep changes minimal and focused.
- Default to clarity over cleverness in robotics math.
- Coordinate with users before running long training jobs.
