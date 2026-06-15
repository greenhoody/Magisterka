from __future__ import annotations

from datetime import datetime
from pathlib import Path


def build_unique_run_paths(
    out_dir_root: str | Path,
    policy_module: str,
    script_path: str | Path,
    env_size: int,
    metrics_prefix: str,
    checkpoint_prefix: str,
) -> tuple[str, Path, Path, Path, Path, str]:
    script_stem = Path(script_path).stem
    base_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    attempt = 1

    while True:
        if attempt == 1:
            run_tag = base_tag
        else:
            run_tag = f"{base_tag}_v{attempt:02d}"

        run_name = f"{policy_module}__{script_stem}__{run_tag}"
        out_dir = Path(out_dir_root) / run_name / f"botbowl-{env_size}"

        if not out_dir.exists():
            metrics_path = out_dir / f"{metrics_prefix}__{run_tag}.csv"
            best_ckpt = out_dir / f"{checkpoint_prefix}__{run_tag}_best.pt"
            final_ckpt = out_dir / f"{checkpoint_prefix}__{run_tag}_final.pt"
            return run_name, out_dir, metrics_path, best_ckpt, final_ckpt, run_tag

        attempt += 1
