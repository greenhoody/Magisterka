#!/usr/bin/env python3
"""
Run multiple *_random_checkpoint_training.py scripts concurrently.

This behaves like train_all_checkpoint_agents.py but allows up to N jobs to run
in parallel (default: 3). Each child process receives BOTBOWL_ENV_SIZE=<value>
so we can shrink arenas (default 5) and fit more concurrent trainings on one GPU.
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

from training_env import ENV_SIZE_ENV

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable or "python"


@dataclass(frozen=True)
class TrainingJob:
    key: str
    script: str
    description: str

    def command(self) -> List[str]:
        return [PYTHON, "-u", self.script]


@dataclass
class RunningJob:
    job: TrainingJob
    index: int
    process: subprocess.Popen
    started_at: float


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
        description="Run multiple training scripts in parallel."
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
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum number of concurrent jobs. Default: %(default)s",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        default=5,
        help="BOTBOWL_ENV_SIZE override passed to child processes. Default: %(default)s",
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


def launch_job(
    job: TrainingJob,
    index: int,
    total: int,
    env_size: int,
) -> RunningJob:
    cmd = job.command()
    pretty_cmd = " ".join(cmd)
    print(
        f"\n[{index}/{total}] Starting {job.description}\n"
        f"    Script : {job.script}\n"
        f"    Command: {pretty_cmd}\n"
        f"    {ENV_SIZE_ENV}={env_size}"
    )
    env = os.environ.copy()
    env[ENV_SIZE_ENV] = str(env_size)
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env)
    return RunningJob(job=job, index=index, process=process, started_at=time.perf_counter())


def wait_for_any(running: List[RunningJob]) -> tuple[RunningJob, float]:
    while True:
        for entry in running:
            ret = entry.process.poll()
            if ret is None:
                continue
            running.remove(entry)
            if ret != 0:
                raise SystemExit(f"{entry.job.key} failed with exit code {ret}")
            elapsed = time.perf_counter() - entry.started_at
            print(
                f"[{entry.job.key}] Finished in {format_duration(elapsed)} "
                f"(slot freed, {len(running)} job(s) still running)"
            )
            return entry, elapsed
        time.sleep(0.5)


def run_parallel(
    jobs: List[TrainingJob],
    max_parallel: int,
    env_size: int,
) -> None:
    if max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")

    total = len(jobs)
    running: List[RunningJob] = []
    job_iter = iter(enumerate(jobs, start=1))
    completed = 0
    total_elapsed = 0.0

    try:
        while completed < total:
            while len(running) < max_parallel:
                try:
                    idx, job = next(job_iter)
                except StopIteration:
                    break
                running.append(launch_job(job, idx, total, env_size))
            if not running:
                break
            finished, elapsed = wait_for_any(running)
            completed += 1
            total_elapsed += elapsed
            remaining = total - completed
            if remaining:
                avg = total_elapsed / completed
                eta = avg * remaining
                print(
                    f"    Completed {completed}/{total} | "
                    f"Estimated time for remaining {remaining}: {format_duration(eta)}"
                )
    finally:
        for entry in running:
            entry.process.terminate()


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

    if args.dry_run:
        print(
            f"\n(dry-run) Would launch up to {args.max_parallel} job(s) at a time "
            f"with {ENV_SIZE_ENV}={args.env_size}"
        )
        for idx, job in enumerate(jobs, start=1):
            print(
                f"[{idx}/{len(jobs)}] {job.description} "
                f"(Command: {' '.join(job.command())})"
            )
        return

    run_parallel(jobs, args.max_parallel, args.env_size)
    print("\nAll jobs completed.")


if __name__ == "__main__":
    main()
