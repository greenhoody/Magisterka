#!/usr/bin/env python3
"""
Run only the PPO ResNet and PPO Spatial Inception random-to-checkpoint training
jobs, one after the other, with simple timing and optional dry-run support.

Usage:
    python train_ppo_checkpoint_agents.py
    python train_ppo_checkpoint_agents.py --dry-run
"""

from __future__ import annotations

import argparse
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


AVAILABLE_JOBS: Dict[str, TrainingJob] = {
    job.key: job
    for job in [
        TrainingJob(
            key="ppo-resnet",
            script="ppo_resnet_random_checkpoint_training.py",
            description="PPO ResNet (random -> checkpoint opponents)",
        ),
        TrainingJob(
            key="ppo-spatial-inception",
            script="ppo_spatial_inception_random_checkpoint_training.py",
            description="PPO Spatial Inception (random -> checkpoint opponents)",
        ),
    ]
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO ResNet + Spatial Inception checkpoint agents sequentially."
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=AVAILABLE_JOBS.keys(),
        default=list(AVAILABLE_JOBS.keys()),
        help="Optional subset of PPO jobs to run (default: both).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Limit how many jobs from the (ordered) selection should run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected commands without launching training.",
    )
    return parser.parse_args(argv)


def resolve_jobs(selected_keys: Iterable[str], max_jobs: int | None) -> List[TrainingJob]:
    jobs = [AVAILABLE_JOBS[k] for k in selected_keys]
    if max_jobs is not None:
        jobs = jobs[: max(0, max_jobs)]
    return jobs


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
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
    jobs = resolve_jobs(args.jobs, args.max_jobs)
    if not jobs:
        print("No PPO jobs selected. Nothing to do.")
        return

    print("Planned PPO checkpoint training jobs:")
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
        print(f"\nAll selected PPO jobs completed in {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
