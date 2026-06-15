from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


DEFAULT_RUN = "a2c_manifold_dynamic_hyperconnection_11x11__train_drefsante_curriculum__20260518_201654"
DEFAULT_CSV = (
    Path("runs")
    / DEFAULT_RUN
    / "botbowl-11"
    / "drefsante_curriculum_metrics__20260518_201654.csv"
)
DEFAULT_OUT_DIR = Path("wykresy") / DEFAULT_RUN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RL training plots for the Drefsante curriculum run."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("rl", "imitation", "all"),
        default=["all"],
        help="Which plot group to generate.",
    )
    return parser.parse_args()


def read_series(csv_path: Path, phase: str, column: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("phase") != phase:
                continue
            x_raw = row.get("timesteps", "")
            y_raw = row.get(column, "")
            if not x_raw or not y_raw:
                continue
            try:
                points.append((float(x_raw), float(y_raw)))
            except ValueError:
                continue
    if not points:
        raise ValueError(f"No points found for phase={phase!r}, column={column!r}")
    return points


def nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if vmin == vmax:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + step * i for i in range(count)]


def format_tick(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) < 10 and value != int(value):
        return f"{value:.2f}"
    return f"{value:.0f}"


def polyline_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def draw_svg(
    points: list[tuple[float, float]],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    width, height = 1400, 820
    left, right = 118, 44
    top, bottom = 92, 104
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    if ymin == ymax:
        pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
        ymin -= pad
        ymax += pad
    else:
        pad = (ymax - ymin) * 0.08
        ymin -= pad
        ymax += pad

    def map_x(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def map_y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    mapped = [(map_x(x), map_y(y)) for x, y in points]
    x_ticks = nice_ticks(xmin, xmax)
    y_ticks = nice_ticks(ymin, ymax)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="48" fill="#2a3038" font-family="Arial, sans-serif" font-size="34" font-weight="700">{html.escape(title)}</text>',
    ]

    for tick in x_ticks:
        x = map_x(tick)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e0e5eb" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 36}" text-anchor="middle" fill="#2a3038" font-family="Arial, sans-serif" font-size="20">{format_tick(tick)}</text>')

    for tick in y_ticks:
        y = map_y(tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e0e5eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 14}" y="{y + 7:.2f}" text-anchor="end" fill="#2a3038" font-family="Arial, sans-serif" font-size="20">{format_tick(tick)}</text>')

    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#2a3038" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#2a3038" stroke-width="2"/>',
            f'<polyline fill="none" stroke="#1969b4" stroke-width="3" points="{polyline_points(mapped)}"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 42}" text-anchor="middle" fill="#2a3038" font-family="Arial, sans-serif" font-size="24">steps</text>',
            f'<text x="24" y="{top + plot_h / 2}" fill="#2a3038" font-family="Arial, sans-serif" font-size="24">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    requested = set(args.plots)
    if "all" in requested:
        requested = {"rl", "imitation"}

    if "rl" in requested:
        draw_svg(
            read_series(args.csv, "reinforcement", "win_rate_total"),
            "Win rate vs RL steps",
            "win rate",
            args.out_dir / "win_rate_vs_rl_steps.svg",
        )
        draw_svg(
            read_series(args.csv, "reinforcement", "mean_episode_return_20"),
            "Reward vs RL steps",
            "mean reward (20 episodes)",
            args.out_dir / "reward_vs_rl_steps.svg",
        )

    if "imitation" in requested:
        draw_svg(
            read_series(args.csv, "imitation", "imitation_loss"),
            "Imitation loss vs imitation steps",
            "imitation loss",
            args.out_dir / "imitation_loss_vs_steps.svg",
        )
        draw_svg(
            read_series(args.csv, "imitation", "expert_recorded"),
            "Expert samples vs imitation steps",
            "expert samples",
            args.out_dir / "expert_samples_vs_imitation_steps.svg",
        )

    print(f"Saved plots to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
