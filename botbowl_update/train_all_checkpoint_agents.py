#!/usr/bin/env python3
"""
Run every random-to-checkpoint training script (A2C + PPO by default) one after
another, showing how long each job takes and the cumulative wall-clock time.

Usage examples:
    python train_all_checkpoint_agents.py
    python train_all_checkpoint_agents.py --only a2c ppo --max-jobs 1
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable or "python"


@dataclass(frozen=True)
class TrainingJob:
    key: str
    script: str
    description: str

    def command(self) -> List[str]:
        return [PYTHON, "-u", self.script]


def discover_jobs() -> Dict[str, TrainingJob]:
    pattern = os.path.join(REPO_ROOT, "*_random_checkpoint_training.py")
    job_map: Dict[str, TrainingJob] = {}
    for script_path in sorted(glob.glob(pattern)):
        script = os.path.basename(script_path)
        base = script.removesuffix("_random_checkpoint_training.py")
        key = base.replace("_", "-")
        description = (
            f"{base.replace('_', ' ').title()} (random -> checkpoint opponents)"
        )
        job_map[key] = TrainingJob(key=key, script=script, description=description)
    return job_map


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple training scripts sequentially with timing info."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="Subset of jobs to run (keys listed at startup).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optionally limit how many jobs (from the selected list) will run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without launching training.",
    )
    return parser.parse_args(argv)


def resolve_jobs(
    available: Dict[str, TrainingJob],
    selected_keys: Iterable[str] | None,
    max_jobs: int | None,
) -> List[TrainingJob]:
    if selected_keys:
        unknown = [k for k in selected_keys if k not in available]
        if unknown:
            raise SystemExit(f"Unknown job keys: {', '.join(unknown)}")
        jobs = [available[k] for k in selected_keys]
    else:
        jobs = list(available.values())

    if max_jobs is not None:
        jobs = jobs[: max(0, max_jobs)]
    return jobs


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def run_job(job: TrainingJob, index: int, total: int, dry_run: bool) -> float:
    cmd = job.command()
    pretty_cmd = " ".join(cmd)
    print(
        f"\n[{index}/{total}] Starting {job.description}\n"
        f"    Script : {job.script}\n"
        f"    Command: {pretty_cmd}"
    )

    if dry_run:
        print("    (dry-run) Skipping execution.")
        return 0.0

    start = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{job.key} failed with exit code {exc.returncode}") from exc
    elapsed = time.perf_counter() - start
    print(f"[{job.key}] Finished in {format_duration(elapsed)}")
    return elapsed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    job_map = discover_jobs()
    if not job_map:
        print("No *_random_checkpoint_training.py scripts found. Nothing to do.")
        return

    jobs = resolve_jobs(job_map, args.only, args.max_jobs)
    if not jobs:
        print("No jobs selected. Nothing to do.")
        return

    print("Planned training jobs:")
    for job in jobs:
        print(f"  - {job.key}: {job.description} [{job.script}]")

    total_elapsed = 0.0
    for idx, job in enumerate(jobs, start=1):
        elapsed = run_job(job, idx, len(jobs), args.dry_run)
        total_elapsed += elapsed
        if not args.dry_run:
            remaining = len(jobs) - idx
            if remaining:
                avg = total_elapsed / idx
                eta = avg * remaining
                print(
                    f"    Total elapsed: {format_duration(total_elapsed)} | "
                    f"Estimated remaining for {remaining} more job(s): {format_duration(eta)}"
                )

    if not args.dry_run:
        print(f"\nAll jobs completed in {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
