# Repository Guidelines

## Project Structure & Module Organization
- `botbowl/` Python package with gameplay engine (`core`), AI helpers (`ai`), data assets, and the Flask web UI under `web`.
- `botbowl/tests/` extensive pytest suite grouped by domain (gameplay, framework, kickoff, etc.); keep new tests beside related modules.
- `botbowl_update/` custom training code (A2C agents, GPU-ready models, utilities) and lightweight tests for repository-specific features.
- `botbowl_update/tests/` current Torch/JIT regression checks for model saving; mirror this pattern when adding utilities.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` create an isolated environment before installing dependencies.
- `pip install -r botbowl/requirements.txt` fetch core framework dependencies; use `pip install -e botbowl` for editable package work.
- `pytest -q botbowl/tests` run framework verification; use `pytest -q botbowl_update/tests` for repo-specific checks.
- `python examples/server_example.py` (from `botbowl/`) starts the local web UI on port 1234 for manual inspection.
- `python botbowl_update/my_training.py` launches the current GPU-enabled A2C experiment; pass `CUDA_VISIBLE_DEVICES` when pinning hardware.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, PEP 8 naming (`lower_snake_case` for functions, `PascalCase` for classes, constants in `ALL_CAPS`).
- Prefer explicit type hints (see `my_training.py`) and short docstrings that state intent, especially for utilities and wrappers.
- Keep tensors and arrays named by role (`spatial_obs`, `action_mask`) and log GPU-related prints behind guards when adding new scripts.
- When touching shared engines, favor small, testable helpers over inline blocks; ensure imports stay sorted within each file.

## Testing Guidelines
- Primary framework: `pytest`; stick to `assert`-style checks and use fixtures for reusable environments or neural nets.
- Name tests `test_<behavior>.py`; inside, group logically related cases and mark slow GPU scenarios with `@pytest.mark.slow`.
- Maintain deterministic seeds (`torch.manual_seed`, `np.random.seed`) in learning tests and clean up temporary resources.
- Run both framework and update suites before pushing; add regression tests for new agent behaviors or save/load paths.

## Commit & Pull Request Guidelines
- Follow the existing short, descriptive style (`Dodanie plikow do uczenia na gpu`, `Add repository-wide ignore rules`); write in Polish or English but stay consistent within a PR.
- One feature or fix per commit; include relevant test invocations in the message footer (`Tests: pytest botbowl_update/tests`).
- PRs should summarize the change set, reference tracked issues, and attach artifacts when UI or training outputs change.
- Highlight GPU assumptions, new dependencies, and any required data files in the PR description to streamline reviews.

## GPU & Environment Notes
- Torch auto-detects CUDA (`torch.cuda.is_available()` in `my_training.py`); document fallback behavior if adding CPU-only modes.
- Update `botbowl_update/my_gpu_agents.py` inception block parameters carefully—shape mismatches surface late in training, so log tensor sizes when experimenting.
- Custom bots in `botbowl` should remain engine-agnostic; integrate experimental agents through `botbowl_update/` to avoid leaking research code into the core package.
