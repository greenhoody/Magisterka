from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_metrics(csv_path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value is None or value == "" or isinstance(value, list):
                    continue
                try:
                    data.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return data


def discover_metrics_files(base_dir: Path) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    patterns = (
        "runs_small_network*",
        "run_small_network*",
        "runs_non_spatial_hidden_nodes_actor_and_critic_action_space",
    )

    for pattern in patterns:
        for run_dir in sorted(base_dir.glob(pattern)):
            csv_path = run_dir / "botbowl-3" / "training_metrics.csv"
            if csv_path.exists():
                runs[run_dir.name] = csv_path
    if not runs:
        raise SystemExit(
            f"No training metrics found under {base_dir} for expected run folders"
        )
    return runs


def compute_interval_trend(
    updates: List[float], values: List[float], interval: int
) -> tuple[List[float], List[float]]:
    buckets: Dict[int, List[float]] = {}
    for update, value in zip(updates, values):
        bucket_start = (int(update) // interval) * interval
        buckets.setdefault(bucket_start, []).append(value)

    trend_x: List[float] = []
    trend_y: List[float] = []
    for bucket_start in sorted(buckets.keys()):
        bucket_values = buckets[bucket_start]
        trend_x.append(bucket_start + interval / 2)
        trend_y.append(sum(bucket_values) / len(bucket_values))
    return trend_x, trend_y


def plot_metric(
    runs_data: Dict[str, Dict[str, List[float]]],
    metric: str,
    out_file: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    trend_every: int | None = None,
) -> None:
    plt.figure(figsize=(11, 6))
    for run_name, data in runs_data.items():
        if metric not in data:
            continue
        raw_line, = plt.plot(
            data["update"], data[metric], label=run_name, linewidth=1.2, alpha=0.35
        )
        if trend_every:
            trend_x, trend_y = compute_interval_trend(
                updates=data["update"], values=data[metric], interval=trend_every
            )
            plt.plot(
                trend_x,
                trend_y,
                linestyle="--",
                linewidth=2.8,
                marker="o",
                markersize=3,
                color=raw_line.get_color(),
                label=f"{run_name} trend/{trend_every}",
            )

    plt.xlabel("update")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for runs_small_network* experiments"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory containing runs_small_network* folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("wykresy/compare_runs_small_network"),
        help="Output directory for comparison plots",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help=(
            "Optional list of run folder names to compare. "
            "Example: --runs run_a run_b"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = discover_metrics_files(args.base_dir.resolve())
    if args.runs:
        selected_runs = {name: runs[name] for name in args.runs if name in runs}
        missing = [name for name in args.runs if name not in runs]
        if missing:
            raise SystemExit(
                f"Requested runs not found: {', '.join(missing)}. "
                f"Available runs: {', '.join(sorted(runs.keys()))}"
            )
        runs = selected_runs
    runs_data = {run_name: read_metrics(csv_path) for run_name, csv_path in runs.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        runs_data=runs_data,
        metric="win_rate_50",
        out_file=args.out_dir / "compare_win_rate_50.png",
        title="Comparison: Win Rate (last 50 episodes) vs Update",
        ylabel="win_rate_50",
        ylim=(0.0, 1.0),
        trend_every=500,
    )
    plot_metric(
        runs_data=runs_data,
        metric="win_rate_total",
        out_file=args.out_dir / "compare_win_rate_total.png",
        title="Comparison: Total Win Rate vs Update",
        ylabel="win_rate_total",
        ylim=(0.0, 1.0),
    )
    plot_metric(
        runs_data=runs_data,
        metric="mean_td_for_50",
        out_file=args.out_dir / "compare_mean_td_for_50.png",
        title="Comparison: Mean TD For (last 50 episodes) vs Update",
        ylabel="mean_td_for_50",
        trend_every=500,
    )

    print(f"Saved comparison plots to: {args.out_dir.resolve()}")
    print("Generated files:")
    print("- compare_win_rate_50.png")
    print("- compare_win_rate_total.png")
    print("- compare_mean_td_for_50.png")


if __name__ == "__main__":
    main()
