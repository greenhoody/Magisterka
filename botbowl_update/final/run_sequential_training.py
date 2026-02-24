from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import fields
from typing import Dict, Iterable, List


DEFAULT_NETWORKS = [
    "small_network_inception_block",
    "small_network_inception_residual_block",
    "small_network_hyperconnection_inception_block",
    "small_network_dynamic_hyperconnection_inception_block",
]

TRAINING_SCRIPTS = {
    "a2c": "train_small_a2c_mp",
    "ppo": "train_small_ppo_mp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sekwencyjne uruchamianie treningu dla wielu sieci metodami A2C i PPO."
        )
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=DEFAULT_NETWORKS,
        help="Lista modułów sieci (np. small_network_inception_block).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=sorted(TRAINING_SCRIPTS.keys()),
        default=["a2c", "ppo"],
        help="Algorytmy do uruchomienia dla każdej sieci.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Nadpisuje Config.num_steps dla wszystkich uruchomień.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Nadpisuje Config.num_envs dla wszystkich uruchomień.",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        help="Nadpisuje Config.rollout_len dla wszystkich uruchomień.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Nadpisuje Config.seed dla wszystkich uruchomień.",
    )
    parser.add_argument(
        "--out-dir-root",
        type=str,
        help="Nadpisuje Config.out_dir_root dla wszystkich uruchomień.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Przerywa całą sekwencję po pierwszym błędzie.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko wypisuje plan uruchomień bez trenowania.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace, policy_module: str) -> Dict[str, object]:
    overrides: Dict[str, object] = {"policy_module": policy_module}
    if args.num_steps is not None:
        overrides["num_steps"] = args.num_steps
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.rollout_len is not None:
        overrides["rollout_len"] = args.rollout_len
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.out_dir_root is not None:
        overrides["out_dir_root"] = args.out_dir_root
    return overrides


def _config_keys(config_cls) -> set[str]:
    return {field.name for field in fields(config_cls)}


def run_training_module(module_name: str, overrides: Dict[str, object]) -> None:
    module = importlib.import_module(module_name)
    original_config_cls = module.Config
    allowed_keys = _config_keys(original_config_cls)
    invalid_keys = sorted(set(overrides.keys()) - allowed_keys)
    if invalid_keys:
        keys = ", ".join(invalid_keys)
        raise ValueError(f"{module_name}: nieznane pola Config: {keys}")

    def configured_config():
        cfg = original_config_cls()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    module.Config = configured_config
    try:
        module.main()
    finally:
        module.Config = original_config_cls


def planned_runs(
    networks: Iterable[str], algorithms: Iterable[str]
) -> List[tuple[str, str]]:
    runs: List[tuple[str, str]] = []
    for network in networks:
        for algorithm in algorithms:
            runs.append((algorithm, network))
    return runs


def main() -> None:
    args = parse_args()
    runs = planned_runs(args.networks, args.algorithms)
    total = len(runs)

    if total == 0:
        print("Brak uruchomień do wykonania.")
        return

    print(f"Zaplanowano {total} uruchomien.")
    for idx, (algorithm, network) in enumerate(runs, start=1):
        script = TRAINING_SCRIPTS[algorithm]
        print(f"[{idx}/{total}] {algorithm.upper()} | {network} | {script}")

    if args.dry_run:
        print("Tryb dry-run: nic nie uruchomiono.")
        return

    started_at = time.time()
    failures: List[tuple[str, str, str]] = []

    for idx, (algorithm, network) in enumerate(runs, start=1):
        script = TRAINING_SCRIPTS[algorithm]
        overrides = build_overrides(args, policy_module=network)

        print(f"\n=== START [{idx}/{total}] {algorithm.upper()} | {network} ===")
        step_started_at = time.time()
        try:
            run_training_module(script, overrides)
        except Exception as exc:
            elapsed = time.time() - step_started_at
            print(
                f"=== ERROR [{idx}/{total}] {algorithm.upper()} | {network} "
                f"({elapsed:.1f}s): {exc} ==="
            )
            failures.append((algorithm, network, str(exc)))
            if args.stop_on_error:
                break
        else:
            elapsed = time.time() - step_started_at
            print(
                f"=== DONE  [{idx}/{total}] {algorithm.upper()} | {network} "
                f"({elapsed:.1f}s) ==="
            )

    total_elapsed = time.time() - started_at
    print(f"\nCzas calkowity: {total_elapsed:.1f}s")
    if failures:
        print(f"Niepowodzenia: {len(failures)}")
        for algorithm, network, error in failures:
            print(f"- {algorithm.upper()} | {network}: {error}")
        raise SystemExit(1)

    print("Wszystkie uruchomienia zakonczone sukcesem.")


if __name__ == "__main__":
    main()
