from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


TOTAL_STEPS = 5_000_000
WIN_RATE_COLUMN = "win_rate_50"
REWARD_COLUMN = "mean_episode_return_50"
OUTPUT_DIR = Path("wykresy/tournament_thesis_training_curves_20260419_213029")
OUTPUT_FILE = OUTPUT_DIR / "winrate50_reward_vs_steps_inception_constant_residual_ln.png"
WIDTH = 2200
HEIGHT = 1550
LEFT = 150
RIGHT = 70
TOP_1 = 110
PLOT_HEIGHT = 470
TOP_2 = 700
LEGEND_TOP = 1295


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    csv_path: Path
    color: str


SERIES: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        label="Inception A2C",
        csv_path=Path(
            "runs/small_network_inception_block__train_small_a2c_mp__20260414_003608/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_003608.csv"
        ),
        color="#7fb6d6",
    ),
    SeriesSpec(
        label="Inception Constant z Layer Normalization A2C",
        csv_path=Path(
            "runs/small_network_inception_block_controlled__train_small_a2c_mp__20260414_015859/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_015859.csv"
        ),
        color="#ff4d4d",
    ),
    SeriesSpec(
        label="Residual A2C",
        csv_path=Path(
            "runs/small_network_inception_residual_block_no_layer_norm__train_small_a2c_mp__20260414_051142/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_051142.csv"
        ),
        color="#2f6fa3",
    ),
    SeriesSpec(
        label="Residual Constant A2C",
        csv_path=Path(
            "runs/small_network_inception_residual_block_no_layer_norm_controlled__train_small_a2c_mp__20260414_063938/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_063938.csv"
        ),
        color="#175a7d",
    ),
    SeriesSpec(
        label="Hyperconnection A2C",
        csv_path=Path(
            "runs/small_network_hyperconnection_inception_block__train_small_a2c_mp__20260419_024909/"
            "botbowl-3/training_metrics_no_end_rewards__20260419_024909.csv"
        ),
        color="#9ed3e6",
    ),
    SeriesSpec(
        label="Dynamic Hyperconnection A2C",
        csv_path=Path(
            "runs/small_network_dynamic_hyperconnection_inception_block__train_small_a2c_mp__20260414_094408/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_094408.csv"
        ),
        color="#2b86c5",
    ),
    SeriesSpec(
        label="Residual z Layer Normalization A2C",
        csv_path=Path(
            "runs/small_network_inception_residual_block__train_small_a2c_mp__20260414_032840/"
            "botbowl-3/training_metrics_no_end_rewards__20260414_032840.csv"
        ),
        color="#d62728",
    ),
    SeriesSpec(
        label="Inception PPO",
        csv_path=Path(
            "runs/small_network_inception_block__train_small_ppo_mp__20260419_043127/"
            "botbowl-3/training_metrics_no_end_rewards__20260419_043127.csv"
        ),
        color="#8bcf70",
    ),
    SeriesSpec(
        label="Residual PPO",
        csv_path=Path(
            "runs/small_network_inception_residual_block__train_small_ppo_mp__20260419_063339/"
            "botbowl-3/training_metrics_no_end_rewards__20260419_063339.csv"
        ),
        color="#63bd63",
    ),
    SeriesSpec(
        label="Hyperconnection PPO",
        csv_path=Path(
            "runs/small_network_hyperconnection_inception_block__train_small_ppo_mp__20260419_090456/"
            "botbowl-3/training_metrics_no_end_rewards__20260419_090456.csv"
        ),
        color="#46a658",
    ),
    SeriesSpec(
        label="Dynamic Hyperconnection PPO",
        csv_path=Path(
            "runs/small_network_dynamic_hyperconnection_inception_block__train_small_ppo_mp__20260419_120203/"
            "botbowl-3/training_metrics_no_end_rewards__20260419_120203.csv"
        ),
        color="#207a3b",
    ),
)


def read_series(csv_path: Path) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {
        "steps": [],
        "win_rate": [],
        "reward": [],
    }
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["steps"].append(float(row["timesteps"]))
            data["win_rate"].append(float(row[WIN_RATE_COLUMN]))
            data["reward"].append(float(row[REWARD_COLUMN]))
    return data


def require_files(series: Iterable[SeriesSpec]) -> None:
    missing = [str(spec.csv_path) for spec in series if not spec.csv_path.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files:\n" + "\n".join(missing))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def hex_to_rgba(hex_color: str, alpha: int = 178) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def format_steps(value: float) -> str:
    if abs(value) < 1e-9:
        return "0"
    return f"{value / 1_000_000:.0f}M"


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def draw_vertical_text(
    image: Image.Image,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    scratch = Image.new("RGBA", (220, 70), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    width, height = text_size(scratch_draw, text, font)
    scratch_draw.text(((220 - width) / 2, (70 - height) / 2), text, font=font, fill=fill)
    rotated = scratch.rotate(90, expand=True)
    image.alpha_composite(
        rotated,
        (
            int(center[0] - rotated.width / 2),
            int(center[1] - rotated.height / 2),
        ),
    )


def map_point(
    step: float,
    value: float,
    top: int,
    min_y: float,
    max_y: float,
) -> tuple[float, float]:
    plot_width = WIDTH - LEFT - RIGHT
    x = LEFT + (step / TOTAL_STEPS) * plot_width
    if max_y == min_y:
        y = top + PLOT_HEIGHT / 2
    else:
        y = top + (max_y - value) / (max_y - min_y) * PLOT_HEIGHT
    return x, y


def draw_axes(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    top: int,
    title: str,
    y_label: str,
    min_y: float,
    max_y: float,
    fonts: dict[str, ImageFont.ImageFont],
    show_x_label: bool,
) -> None:
    plot_width = WIDTH - LEFT - RIGHT
    bottom = top + PLOT_HEIGHT
    axis_color = "#4b4b4b"
    grid_color = "#dddddd"
    text_color = "#303030"

    draw.text((LEFT, top - 70), title, font=fonts["title"], fill=text_color)
    draw.line((LEFT, top, LEFT, bottom), fill=axis_color, width=2)
    draw.line((LEFT, bottom, LEFT + plot_width, bottom), fill=axis_color, width=2)

    for tick in range(0, TOTAL_STEPS + 1, 1_000_000):
        x = LEFT + (tick / TOTAL_STEPS) * plot_width
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((x, bottom, x, bottom + 8), fill=axis_color, width=2)
        label = format_steps(tick)
        label_width, _ = text_size(draw, label, fonts["small"])
        draw.text((x - label_width / 2, bottom + 15), label, font=fonts["small"], fill=text_color)

    if min_y == 0.0 and max_y == 1.0:
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        y_format = "{:.2f}"
    else:
        y_ticks = [min_y + (max_y - min_y) * i / 5 for i in range(6)]
        y_format = "{:.2f}"

    for value in y_ticks:
        _, y = map_point(0, value, top, min_y, max_y)
        draw.line((LEFT, y, LEFT + plot_width, y), fill=grid_color, width=1)
        draw.line((LEFT - 8, y, LEFT, y), fill=axis_color, width=2)
        label = y_format.format(value)
        label_width, label_height = text_size(draw, label, fonts["small"])
        draw.text(
            (LEFT - label_width - 16, y - label_height / 2),
            label,
            font=fonts["small"],
            fill=text_color,
        )

    draw_vertical_text(image, (48, top + PLOT_HEIGHT / 2), y_label, fonts["axis"], text_color)
    if show_x_label:
        draw_centered_text(
            draw,
            (LEFT + plot_width / 2, bottom + 66),
            "Steps",
            font=fonts["axis"],
            fill=text_color,
        )


def draw_lines(
    image: Image.Image,
    loaded: list[tuple[SeriesSpec, dict[str, list[float]]]],
    top: int,
    metric: str,
    min_y: float,
    max_y: float,
) -> None:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    for spec, data in loaded:
        points = [
            map_point(step, value, top, min_y, max_y)
            for step, value in zip(data["steps"], data[metric])
        ]
        if len(points) > 1:
            draw.line(points, fill=hex_to_rgba(spec.color), width=4, joint="curve")
    image.alpha_composite(overlay)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    series: tuple[SeriesSpec, ...],
    fonts: dict[str, ImageFont.ImageFont],
) -> None:
    columns = 3
    column_width = (WIDTH - LEFT - RIGHT) / columns
    row_height = 48
    line_length = 58

    for idx, spec in enumerate(series):
        col = idx % columns
        row = idx // columns
        x = LEFT + col * column_width
        y = LEGEND_TOP + row * row_height
        draw.line((x, y + 12, x + line_length, y + 12), fill=spec.color, width=5)
        draw.text(
            (x + line_length + 16, y),
            spec.label,
            font=fonts["legend"],
            fill="#303030",
        )


def main() -> None:
    require_files(SERIES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = [(spec, read_series(spec.csv_path)) for spec in SERIES]
    min_reward = min(min(data["reward"]) for _, data in loaded)
    max_reward = max(max(data["reward"]) for _, data in loaded)

    fonts = {
        "title": load_font(36, bold=True),
        "axis": load_font(25),
        "small": load_font(22),
        "legend": load_font(23),
    }

    image = Image.new("RGBA", (WIDTH, HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(image)

    draw_axes(image, draw, TOP_1, "Win rate vs steps", "Win rate", 0.0, 1.0, fonts, False)
    draw_axes(image, draw, TOP_2, "Reward vs steps", "Reward", min_reward, max_reward, fonts, True)
    draw_lines(image, loaded, TOP_1, "win_rate", 0.0, 1.0)
    draw_lines(image, loaded, TOP_2, "reward", min_reward, max_reward)
    draw_legend(draw, SERIES, fonts)

    image.convert("RGB").save(OUTPUT_FILE)

    print(f"Saved plot to: {OUTPUT_FILE.resolve()}")
    print(f"Reward range: {min_reward:.6f} to {max_reward:.6f}")


if __name__ == "__main__":
    main()
