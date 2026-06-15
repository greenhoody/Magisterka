from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

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


def infer_label(csv_path: Path) -> str:
    """
    Infer a readable experiment label from the standard runs/<run>/<env>/file.csv layout.
    """
    if csv_path.parent.name.startswith("botbowl-") and csv_path.parent.parent.name:
        return csv_path.parent.parent.name
    return csv_path.stem


def make_plots(
    data: Dict[str, List[float]],
    out_dir: Path,
    label: str,
    win_rate_title: str,
    mean_reward_title: str,
    file_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    updates = data["update"]

    plt.figure(figsize=(10, 5))
    plt.plot(updates, data["win_rate_total"], label="win_rate_total", linewidth=2)
    plt.plot(updates, data["win_rate_50"], label="win_rate_50", linewidth=2)
    plt.xlabel("update")
    plt.ylabel("win rate")
    plt.title(win_rate_title)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_win_rate_vs_update.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        updates,
        data["mean_episode_return_50"],
        label="mean_episode_return_50",
        linewidth=2,
    )
    plt.xlabel("update")
    plt.ylabel("mean reward")
    plt.title(mean_reward_title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{file_prefix}_mean_reward_vs_update.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate win rate and mean reward plots from training metrics CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to training metrics CSV file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated PNG plots.",
    )
    parser.add_argument(
        "--label",
        type=str,
        help="Optional label shown in plot titles. Defaults to inferred run name.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help=(
            "Base title for both plots. "
            "Example: 'Residual PPO botbowl-3'."
        ),
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        help=(
            "Optional prefix for output PNG filenames. "
            "Defaults to a filesystem-safe version of the label."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file does not exist: {args.csv}")

    data = read_metrics(args.csv)
    required = {
        "update",
        "win_rate_total",
        "win_rate_50",
        "mean_episode_return_50",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")

    label = args.label or infer_label(args.csv)
    title_base = args.title or label
    file_prefix = args.file_prefix or "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label
    ).strip("_")
    if not file_prefix:
        file_prefix = "plot"

    make_plots(
        data,
        args.out_dir,
        label,
        win_rate_title=f"Win Rate vs Update - {title_base}",
        mean_reward_title=f"Mean Reward vs Update - {title_base}",
        file_prefix=file_prefix,
    )
    print(f"Saved plots to: {args.out_dir.resolve()}")
    print(f"Label: {label}")
    print(f"Title base: {title_base}")
    print("Generated files:")
    print(f"- {file_prefix}_win_rate_vs_update.png")
    print(f"- {file_prefix}_mean_reward_vs_update.png")


if __name__ == "__main__":
    main()
