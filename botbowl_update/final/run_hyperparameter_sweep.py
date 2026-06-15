from __future__ import annotations

import argparse
import ast
import itertools
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from run_sequential_training import TRAINING_SCRIPTS, run_training_module


PER_MATCH_TRAINING_SCRIPTS = {
    "a2c": "train_small_a2c_mp_per_match",
    "ppo": "train_small_ppo_mp_per_match",
}

TRAINER_FAMILIES = {
    "standard": TRAINING_SCRIPTS,
    "per-match": PER_MATCH_TRAINING_SCRIPTS,
}

DEFAULT_SWEEP_NETWORKS = [
    "small_network_inception_block_controlled",
]

RESERVED_OVERRIDE_KEYS = {"policy_module", "policy_class", "out_dir_root"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uruchamia sweep hiperparametrow dla wybranych sieci i algorytmow. "
            "Kazdy wpis --grid tworzy liste wartosci, z ktorych budowany jest iloczyn kartezjanski."
        )
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=DEFAULT_SWEEP_NETWORKS,
        help="Lista modulow sieci do trenowania.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["a2c", "ppo"],
        default=["a2c"],
        help="Algorytmy do uruchomienia.",
    )
    parser.add_argument(
        "--trainer",
        choices=sorted(TRAINER_FAMILIES.keys()),
        default="standard",
        help="Rodzina skryptow treningowych.",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        help="Ustawia BOTBOWL_ENV_SIZE na czas treningu.",
    )
    parser.add_argument(
        "--sweep-name",
        default="hyperparameter_sweep",
        help="Nazwa eksperymentu uzywana w katalogu wynikowym.",
    )
    parser.add_argument(
        "--out-dir-root",
        default="runs_sweeps",
        help="Katalog bazowy dla wynikow sweepa.",
    )
    parser.add_argument(
        "--set",
        dest="fixed_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Stala nadpiska Config dla wszystkich uruchomien.",
    )
    parser.add_argument(
        "--grid",
        dest="grid_overrides",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help="Lista wartosci testowych dla wybranego pola Config.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Przerywa sweep po pierwszym bledzie.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wypisuje plan bez uruchamiania treningu.",
    )
    return parser.parse_args()


@contextmanager
def temporary_env(var_name: str, value: str | None) -> Iterator[None]:
    previous = os.environ.get(var_name)
    if value is None:
        yield
        return

    os.environ[var_name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(var_name, None)
        else:
            os.environ[var_name] = previous


def script_names_for(args: argparse.Namespace) -> List[str]:
    scripts = TRAINER_FAMILIES[args.trainer]
    return [scripts[algorithm] for algorithm in args.algorithms]


def load_config_defaults(module_name: str) -> Dict[str, Any]:
    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku modulu: {module_path}")

    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            defaults: Dict[str, Any] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    defaults[stmt.target.id] = infer_exemplar(stmt.annotation, stmt.value)
            if defaults:
                return defaults
            break

    raise ValueError(f"Nie udalo sie odczytac klasy Config z {module_path}")


def infer_exemplar(annotation: ast.expr | None, value: ast.expr | None) -> Any:
    annotated = infer_from_annotation(annotation)
    if annotated is not None:
        return annotated

    literal = infer_from_value(value)
    if literal is not None:
        return literal

    return ""


def infer_from_annotation(annotation: ast.expr | None) -> Any:
    if isinstance(annotation, ast.Name):
        if annotation.id == "int":
            return 0
        if annotation.id == "float":
            return 0.0
        if annotation.id == "str":
            return ""
        if annotation.id == "bool":
            return False
    return None


def infer_from_value(value: ast.expr | None) -> Any:
    if value is None:
        return None

    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    if isinstance(value, ast.Call):
        if isinstance(value.func, ast.Attribute) and value.func.attr == "randint":
            return 0
    return None


def parse_key_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Missing key in assignment: {raw}")
    if not value:
        raise ValueError(f"Missing value in assignment: {raw}")
    return key, value


def parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from: {raw}")


def cast_value(raw: str, exemplar: Any) -> Any:
    if isinstance(exemplar, bool):
        return parse_bool(raw)
    if isinstance(exemplar, int) and not isinstance(exemplar, bool):
        return int(raw)
    if isinstance(exemplar, float):
        return float(raw)
    if exemplar is None:
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return parse_bool(raw)
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def validate_override_keys(
    keys: Sequence[str], defaults_by_script: Dict[str, Dict[str, Any]]
) -> None:
    for key in keys:
        if key in RESERVED_OVERRIDE_KEYS:
            raise ValueError(
                f"Pole {key} jest zarzadzane przez skrypt sweepa i nie moze byc nadpisane."
            )
        missing_in = [
            script_name
            for script_name, defaults in defaults_by_script.items()
            if key not in defaults
        ]
        if missing_in:
            joined = ", ".join(missing_in)
            raise ValueError(f"Pole Config '{key}' nie istnieje w: {joined}")


def build_fixed_overrides(
    raw_assignments: Sequence[str], defaults_by_script: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for raw in raw_assignments:
        key, value = parse_key_value(raw)
        validate_override_keys([key], defaults_by_script)
        exemplar = next(iter(defaults_by_script.values()))[key]
        overrides[key] = cast_value(value, exemplar)
    return overrides


def build_grid_values(
    raw_assignments: Sequence[str], defaults_by_script: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Any]]:
    grid_values: Dict[str, List[Any]] = {}
    for raw in raw_assignments:
        key, value = parse_key_value(raw)
        validate_override_keys([key], defaults_by_script)
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise ValueError(f"Lista wartosci dla {key} jest pusta.")
        exemplar = next(iter(defaults_by_script.values()))[key]
        grid_values[key] = [cast_value(item, exemplar) for item in parts]
    return grid_values


def cartesian_grid(grid_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid_values:
        return [{}]

    keys = list(grid_values.keys())
    combinations: List[Dict[str, Any]] = []
    for values in itertools.product(*(grid_values[key] for key in keys)):
        combinations.append(dict(zip(keys, values)))
    return combinations


def slugify_value(value: Any) -> str:
    text = str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "value"


def combo_slug(values: Dict[str, Any]) -> str:
    if not values:
        return "default"
    parts = [f"{key}-{slugify_value(values[key])}" for key in sorted(values)]
    return "__".join(parts)


def planned_runs(
    algorithms: Sequence[str],
    networks: Sequence[str],
    fixed_overrides: Dict[str, Any],
    grid_combinations: Sequence[Dict[str, Any]],
    base_out_dir: Path,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for algorithm in algorithms:
        for network in networks:
            for grid_values in grid_combinations:
                combined_values = dict(fixed_overrides)
                combined_values.update(grid_values)
                slug = combo_slug(combined_values)
                out_dir_root = base_out_dir / algorithm / network / slug
                overrides = {
                    "policy_module": network,
                    **combined_values,
                    "out_dir_root": str(out_dir_root),
                }
                runs.append(
                    {
                        "algorithm": algorithm,
                        "network": network,
                        "grid_values": grid_values,
                        "all_values": combined_values,
                        "slug": slug,
                        "out_dir_root": out_dir_root,
                        "overrides": overrides,
                    }
                )
    return runs


def main() -> None:
    args = parse_args()
    script_map = TRAINER_FAMILIES[args.trainer]
    defaults_by_script = {
        script_name: load_config_defaults(script_name)
        for script_name in script_names_for(args)
    }

    fixed_overrides = build_fixed_overrides(args.fixed_overrides, defaults_by_script)
    grid_values = build_grid_values(args.grid_overrides, defaults_by_script)
    grid_combinations = cartesian_grid(grid_values)

    base_out_dir = Path(args.out_dir_root) / args.sweep_name
    runs = planned_runs(
        algorithms=args.algorithms,
        networks=args.networks,
        fixed_overrides=fixed_overrides,
        grid_combinations=grid_combinations,
        base_out_dir=base_out_dir,
    )

    if not runs:
        print("Brak uruchomien do wykonania.")
        return

    print(f"Zaplanowano {len(runs)} uruchomien sweepa.")
    for idx, run in enumerate(runs, start=1):
        values = ", ".join(f"{key}={value}" for key, value in sorted(run["all_values"].items()))
        script_name = script_map[run["algorithm"]]
        print(
            f"[{idx}/{len(runs)}] {run['algorithm'].upper()} | {run['network']} | "
            f"{script_name} | {values or 'brak nadpisek'}"
        )
        print(f"    out_dir_root={run['out_dir_root']}")

    if args.dry_run:
        print("Tryb dry-run: nic nie uruchomiono.")
        return

    failures: List[tuple[str, str, str, str]] = []
    started_at = time.time()

    with temporary_env(
        "BOTBOWL_ENV_SIZE",
        str(args.env_size) if args.env_size is not None else None,
    ):
        for idx, run in enumerate(runs, start=1):
            script_name = script_map[run["algorithm"]]
            print(
                f"\n=== START [{idx}/{len(runs)}] {run['algorithm'].upper()} | "
                f"{run['network']} | {run['slug']} ==="
            )
            step_started_at = time.time()
            try:
                run_training_module(script_name, run["overrides"])
            except Exception as exc:
                elapsed = time.time() - step_started_at
                print(
                    f"=== ERROR [{idx}/{len(runs)}] {run['algorithm'].upper()} | "
                    f"{run['network']} | {run['slug']} ({elapsed:.1f}s): {exc} ==="
                )
                failures.append(
                    (run["algorithm"], run["network"], run["slug"], str(exc))
                )
                if args.stop_on_error:
                    break
            else:
                elapsed = time.time() - step_started_at
                print(
                    f"=== DONE  [{idx}/{len(runs)}] {run['algorithm'].upper()} | "
                    f"{run['network']} | {run['slug']} ({elapsed:.1f}s) ==="
                )

    total_elapsed = time.time() - started_at
    print(f"\nCzas calkowity sweepa: {total_elapsed:.1f}s")

    if failures:
        print(f"Niepowodzenia: {len(failures)}")
        for algorithm, network, slug, error in failures:
            print(f"- {algorithm.upper()} | {network} | {slug}: {error}")
        raise SystemExit(1)

    print("Sweep zakonczony sukcesem.")


if __name__ == "__main__":
    main()
