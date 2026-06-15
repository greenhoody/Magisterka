from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DTYPE_BYTES = {
    "bool": 1,
    "bool_": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}

APPROX_NDARRAY_HEADER_BYTES = 112


@dataclass(frozen=True)
class ArraySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    itemsize: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def payload_bytes(self) -> int:
        return self.elements * self.itemsize

    @property
    def approx_array_bytes(self) -> int:
        return self.payload_bytes + APPROX_NDARRAY_HEADER_BYTES


@dataclass(frozen=True)
class Layout:
    spatial: ArraySpec
    non_spatial: ArraySpec
    action_mask: ArraySpec
    action_bytes: int
    action_object_bytes: int

    @property
    def payload_bytes_per_sample(self) -> int:
        return (
            self.spatial.payload_bytes
            + self.non_spatial.payload_bytes
            + self.action_mask.payload_bytes
            + self.action_bytes
        )

    @property
    def approximate_buffer_bytes_per_sample(self) -> int:
        return self.payload_bytes_per_sample

    def approximate_preallocated_buffer_bytes(self, samples: int) -> int:
        return self.payload_bytes_per_sample * samples + (4 * APPROX_NDARRAY_HEADER_BYTES)


def parse_shape(value: str) -> tuple[int, ...]:
    parts = [part for part in re.split(r"[xX, ]+", value.strip()) if part]
    if not parts:
        raise argparse.ArgumentTypeError("shape cannot be empty")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid shape '{value}', expected e.g. 44x11x11 or 44,11,11"
        ) from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def dtype_itemsize(dtype: str) -> int:
    normalized = dtype.lower()
    if normalized not in DTYPE_BYTES:
        known = ", ".join(sorted(DTYPE_BYTES))
        raise argparse.ArgumentTypeError(f"unknown dtype '{dtype}'. Known: {known}")
    return DTYPE_BYTES[normalized]


def format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def build_array_spec(name: str, shape: tuple[int, ...], dtype: str) -> ArraySpec:
    return ArraySpec(
        name=name,
        shape=shape,
        dtype=dtype,
        itemsize=dtype_itemsize(dtype),
    )


def infer_layout_from_env(
    env_size: int,
    pathfinding: bool,
    spatial_dtype: str,
    non_spatial_dtype: str,
    action_mask_dtype: str,
    action_bytes: int,
    use_env_dtypes: bool,
) -> Layout:
    from botbowl.ai.env import BotBowlEnv, EnvConf

    env = BotBowlEnv(EnvConf(size=env_size, pathfinding=pathfinding))
    try:
        spatial, non_spatial, action_mask = env.reset()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return Layout(
        spatial=build_array_spec(
            "spatial",
            tuple(spatial.shape),
            str(spatial.dtype) if use_env_dtypes else spatial_dtype,
        ),
        non_spatial=build_array_spec(
            "non_spatial",
            tuple(non_spatial.shape),
            str(non_spatial.dtype) if use_env_dtypes else non_spatial_dtype,
        ),
        action_mask=build_array_spec(
            "action_mask",
            tuple(action_mask.shape),
            str(action_mask.dtype) if use_env_dtypes else action_mask_dtype,
        ),
        action_bytes=action_bytes,
        action_object_bytes=sys.getsizeof(int(0)),
    )


def build_manual_layout(args: argparse.Namespace) -> Layout:
    missing = [
        name
        for name in ("spatial_shape", "non_spatial_shape", "action_mask_shape")
        if getattr(args, name) is None
    ]
    if missing:
        options = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(
            "Cannot infer BotBowl observation shapes in this Python environment. "
            f"Provide {options}, or run the script from an environment with botbowl installed."
        )

    return Layout(
        spatial=build_array_spec("spatial", args.spatial_shape, args.spatial_dtype),
        non_spatial=build_array_spec(
            "non_spatial", args.non_spatial_shape, args.non_spatial_dtype
        ),
        action_mask=build_array_spec(
            "action_mask", args.action_mask_shape, args.action_mask_dtype
        ),
        action_bytes=args.action_bytes,
        action_object_bytes=sys.getsizeof(int(0)),
    )


def last_metrics_row(path: Path) -> Optional[dict[str, str]]:
    with path.open(newline="") as f:
        last_row = None
        for row in csv.DictReader(f):
            last_row = row
    return last_row


def int_from_row(row: Optional[dict[str, str]], key: str) -> Optional[int]:
    if not row:
        return None
    value = row.get(key, "")
    if value is None or value == "":
        return None
    return int(float(value))


def choose_sample_count(args: argparse.Namespace, row: Optional[dict[str, str]]) -> int:
    if args.samples is not None:
        return args.samples

    metrics_samples = int_from_row(row, "samples")
    if metrics_samples is not None:
        return metrics_samples

    return args.recent_buffer_size + args.archive_buffer_size


def print_array(spec: ArraySpec) -> None:
    shape = "x".join(str(dim) for dim in spec.shape)
    print(
        f"{spec.name:12} shape={shape:<16} dtype={spec.dtype:<8} "
        f"payload/sample={format_bytes(spec.payload_bytes)}"
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how much memory imitation-learning samples occupy in "
            "train_drefsante_curriculum.py's ImitationBuffer."
        )
    )
    parser.add_argument(
        "--env-size",
        type=int,
        default=11,
        help="BotBowl environment size used when probing observation shapes.",
    )
    parser.add_argument(
        "--pathfinding",
        action="store_true",
        help="Use EnvConf(pathfinding=True) while probing BotBowl shapes.",
    )
    parser.add_argument(
        "--no-env-probe",
        action="store_true",
        help="Skip BotBowl import/probing and use manually supplied shapes.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Optional training metrics CSV. Uses the last row's samples value.",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        help="Number of samples to estimate. Overrides --metrics-csv.",
    )
    parser.add_argument(
        "--recent-buffer-size",
        type=positive_int,
        default=20_000,
        help="Fallback sample count when neither --samples nor --metrics-csv is set.",
    )
    parser.add_argument(
        "--archive-buffer-size",
        type=positive_int,
        default=0,
        help="Added to fallback sample count for older/experimental archive buffers.",
    )
    parser.add_argument("--spatial-shape", type=parse_shape)
    parser.add_argument("--non-spatial-shape", type=parse_shape)
    parser.add_argument("--action-mask-shape", type=parse_shape)
    parser.add_argument("--spatial-dtype", default="float32", type=str)
    parser.add_argument("--non-spatial-dtype", default="float32", type=str)
    parser.add_argument("--action-mask-dtype", default="bool", type=str)
    parser.add_argument(
        "--action-bytes",
        type=positive_int,
        default=8,
        help="Raw numeric bytes for one action index; Python object overhead is reported separately.",
    )
    parser.add_argument(
        "--use-env-dtypes",
        action="store_true",
        help="Estimate raw BotBowlEnv dtypes instead of ImitationBuffer storage dtypes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_row = last_metrics_row(args.metrics_csv) if args.metrics_csv else None

    if args.no_env_probe:
        layout = build_manual_layout(args)
        source = "manual arguments"
    else:
        try:
            layout = infer_layout_from_env(
                env_size=args.env_size,
                pathfinding=args.pathfinding,
                spatial_dtype=args.spatial_dtype,
                non_spatial_dtype=args.non_spatial_dtype,
                action_mask_dtype=args.action_mask_dtype,
                action_bytes=args.action_bytes,
                use_env_dtypes=args.use_env_dtypes,
            )
            source = (
                f"BotBowlEnv(size={args.env_size}, pathfinding={args.pathfinding})"
                + (" raw dtypes" if args.use_env_dtypes else " buffer dtypes")
            )
        except Exception as exc:
            print(f"BotBowl probe failed: {exc}", file=sys.stderr)
            layout = build_manual_layout(args)
            source = "manual arguments"

    samples = choose_sample_count(args, metrics_row)
    payload_total = layout.payload_bytes_per_sample * samples
    buffer_total = layout.approximate_preallocated_buffer_bytes(samples)

    print(f"Shape source: {source}")
    print_array(layout.spatial)
    print_array(layout.non_spatial)
    print_array(layout.action_mask)
    print(f"action_idx    raw/sample={format_bytes(layout.action_bytes)}")
    print()
    print(f"Samples estimated: {samples:,}")
    print(
        "Raw tensor/array payload: "
        f"{format_bytes(layout.payload_bytes_per_sample)} per sample, "
        f"{format_bytes(payload_total)} total"
    )
    print(
        "Approx. preallocated ImitationBuffer RAM: "
        f"{format_bytes(layout.approximate_buffer_bytes_per_sample)} per sample, "
        f"{format_bytes(buffer_total)} total"
    )
    print(
        "Approximation assumes one contiguous NumPy array for each buffer field "
        "(spatial, non_spatial, action_mask, actions)."
    )

    total_recorded = int_from_row(metrics_row, "total_recorded_samples")
    if total_recorded is not None and total_recorded != samples:
        recorded_payload = layout.payload_bytes_per_sample * total_recorded
        recorded_buffer = layout.approximate_preallocated_buffer_bytes(total_recorded)
        print()
        print(
            "If every recorded sample were retained instead of reservoir-replaced: "
            f"{total_recorded:,} samples -> payload {format_bytes(recorded_payload)}, "
            f"approx RAM {format_bytes(recorded_buffer)}"
        )


if __name__ == "__main__":
    main()
