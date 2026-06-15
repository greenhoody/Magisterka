from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_metrics(csv_path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    data.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return data


def parse_series(items: List[str]) -> List[Tuple[str, Path]]:
    series: List[Tuple[str, Path]] = []
    for item in items:
        if "=" in item:
            label, csv_path = item.split("=", 1)
        else:
            csv_path = item
            label = Path(csv_path).stem
        series.append((label, Path(csv_path)))
    return series


def plot_metric(
    series_data: List[Tuple[str, Dict[str, List[float]]]],
    metric: str,
    out_file: Path,
    title: str,
) -> None:
    plt.figure(figsize=(11, 6))

    for label, data in series_data:
        plt.plot(
            data["update"],
            data[metric],
            label=label,
            linewidth=2.2,
            alpha=0.5,
        )

    plt.xlabel("update")
    plt.ylabel(metric)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot win rate from multiple training metrics CSV files on one chart."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        required=True,
        help=(
            "Series definitions. Use either PATH or LABEL=PATH. "
            "Example: --series a2c=run1.csv ppo=run2.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated plots.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Win Rate Comparison",
        help="Base title for the comparison plot(s).",
    )
    parser.add_argument(
        "--metric",
        choices=["win_rate_50", "win_rate_total", "both"],
        default="win_rate_50",
        help="Which win rate metric to plot.",
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default="compare",
        help="Prefix for output PNG filenames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed_series = parse_series(args.series)

    series_data: List[Tuple[str, Dict[str, List[float]]]] = []
    required_common = {"update"}

    for label, csv_path in parsed_series:
        if not csv_path.exists():
            raise SystemExit(f"CSV file does not exist: {csv_path}")
        data = read_metrics(csv_path)
        missing_common = sorted(required_common - set(data.keys()))
        if missing_common:
            raise SystemExit(
                f"Missing required columns in CSV {csv_path}: {', '.join(missing_common)}"
            )
        series_data.append((label, data))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.metric in {"win_rate_50", "both"}:
        for label, data in series_data:
            if "win_rate_50" not in data:
                raise SystemExit(f"Missing required column 'win_rate_50' in series: {label}")
        plot_metric(
            series_data=series_data,
            metric="win_rate_50",
            out_file=args.out_dir / f"{args.file_prefix}_win_rate_50.png",
            title=f"{args.title} - win_rate_50",
        )

    if args.metric in {"win_rate_total", "both"}:
        for label, data in series_data:
            if "win_rate_total" not in data:
                raise SystemExit(
                    f"Missing required column 'win_rate_total' in series: {label}"
                )
        plot_metric(
            series_data=series_data,
            metric="win_rate_total",
            out_file=args.out_dir / f"{args.file_prefix}_win_rate_total.png",
            title=f"{args.title} - win_rate_total",
        )

    print(f"Saved comparison plots to: {args.out_dir.resolve()}")
    if args.metric == "both":
        print(f"- {args.file_prefix}_win_rate_50.png")
        print(f"- {args.file_prefix}_win_rate_total.png")
    elif args.metric == "win_rate_50":
        print(f"- {args.file_prefix}_win_rate_50.png")
    else:
        print(f"- {args.file_prefix}_win_rate_total.png")


if __name__ == "__main__":
    main()
