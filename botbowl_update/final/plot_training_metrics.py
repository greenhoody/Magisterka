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
                    # Skip non-numeric columns like "top_actions".
                    continue
    return data


def make_plots(data: Dict[str, List[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    updates = data["update"]

    # 1) Najważniejszy wykres: win rate od update
    plt.figure(figsize=(10, 5))
    plt.plot(updates, data["win_rate_total"], label="win_rate_total", linewidth=2)
    plt.plot(updates, data["win_rate_50"], label="win_rate_50", linewidth=2)
    plt.xlabel("update")
    plt.ylabel("win rate")
    plt.title("Win Rate vs Update")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "win_rate_vs_update.png", dpi=150)
    plt.close()

    # 2) Touchdowny
    plt.figure(figsize=(10, 5))
    plt.plot(updates, data["mean_td_for_50"], label="mean_td_for_50", linewidth=2)
    plt.plot(updates, data["mean_td_opponent_50"], label="mean_td_opponent_50", linewidth=2)
    plt.xlabel("update")
    plt.ylabel("touchdowns (mean over last 50 episodes)")
    plt.title("Touchdowns vs Update")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "touchdowns_vs_update.png", dpi=150)
    plt.close()

    # 3) Value loss osobno (ma zwykle większą skalę)
    plt.figure(figsize=(10, 5))
    plt.plot(updates, data["value_loss"], label="value_loss", linewidth=2)
    plt.xlabel("update")
    plt.ylabel("value_loss")
    plt.title("Value Loss vs Update")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "value_loss_vs_update.png", dpi=150)
    plt.close()

    # 4) Policy loss + entropy (podobniejsza skala)
    plt.figure(figsize=(10, 5))
    plt.plot(updates, data["policy_loss"], label="policy_loss", linewidth=2)
    plt.plot(updates, data["policy_entropy"], label="policy_entropy", linewidth=2)
    plt.xlabel("update")
    plt.ylabel("metric value")
    plt.title("Policy Loss and Entropy vs Update")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "policy_loss_entropy_vs_update.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    default_csv = Path(
        "/home/greenhoody/Magisterka/botbowl_update/male_sieci/"
        "runs_small_network_ppo_mp/botbowl-3/training_metrics.csv"
    )
    parser = argparse.ArgumentParser(
        description="Generate training plots from training_metrics.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Path to metrics CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots_from_csv"),
        help="Output directory for PNG plots",
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
        "mean_td_for_50",
        "mean_td_opponent_50",
        "value_loss",
        "policy_loss",
        "policy_entropy",
    }
    missing = sorted(required - set(data.keys()))
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {', '.join(missing)}")

    make_plots(data, args.out_dir)
    print(f"Saved plots to: {args.out_dir.resolve()}")
    print("Generated files:")
    print("- win_rate_vs_update.png")
    print("- touchdowns_vs_update.png")
    print("- value_loss_vs_update.png")
    print("- policy_loss_entropy_vs_update.png")


if __name__ == "__main__":
    main()
