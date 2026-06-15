from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from typing import Iterator, List, Sequence

from run_sequential_training import (
    DEFAULT_NETWORKS,
    TRAINING_SCRIPTS,
    build_overrides,
    run_training_module,
)


PER_MATCH_TRAINING_SCRIPTS = {
    "a2c": "train_small_a2c_mp_per_match",
    "ppo": "train_small_ppo_mp_per_match",
}

TRAINER_FAMILIES = {
    "standard": TRAINING_SCRIPTS,
    "per-match": PER_MATCH_TRAINING_SCRIPTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uruchamia trening wszystkich sieci sekwencyjnie."
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=DEFAULT_NETWORKS,
        help="Lista modulow sieci do trenowania.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["a2c", "ppo"],
        default=["a2c"],
        help="Algorytmy do uruchomienia dla kazdej sieci.",
    )
    parser.add_argument(
        "--trainer",
        choices=sorted(TRAINER_FAMILIES.keys()),
        default="standard",
        help="Wybiera rodzine skryptow treningowych.",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        help="Ustawia BOTBOWL_ENV_SIZE na czas treningu.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Nadpisuje Config.num_steps.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Nadpisuje Config.num_envs.",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        help="Nadpisuje Config.rollout_len.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Nadpisuje Config.seed.",
    )
    parser.add_argument(
        "--out-dir-root",
        type=str,
        help="Nadpisuje Config.out_dir_root.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Przerywa po pierwszym bledzie.",
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


def planned_runs(networks: Sequence[str], algorithms: Sequence[str]) -> List[tuple[str, str]]:
    return [(algorithm, network) for network in networks for algorithm in algorithms]


def main() -> None:
    args = parse_args()
    scripts = TRAINER_FAMILIES[args.trainer]
    runs = planned_runs(args.networks, args.algorithms)

    if not runs:
        print("Brak uruchomien do wykonania.")
        return

    print(f"Zaplanowano {len(runs)} uruchomien.")
    for idx, (algorithm, network) in enumerate(runs, start=1):
        print(f"[{idx}/{len(runs)}] {algorithm.upper()} | {network} | {scripts[algorithm]}")

    if args.dry_run:
        print("Tryb dry-run: nic nie uruchomiono.")
        return

    failures: List[tuple[str, str, str]] = []
    started_at = time.time()

    with temporary_env(
        "BOTBOWL_ENV_SIZE",
        str(args.env_size) if args.env_size is not None else None,
    ):
        for idx, (algorithm, network) in enumerate(runs, start=1):
            script_name = scripts[algorithm]
            overrides = build_overrides(args, policy_module=network)

            print(f"\n=== START [{idx}/{len(runs)}] {algorithm.upper()} | {network} ===")
            step_started_at = time.time()
            try:
                run_training_module(script_name, overrides)
            except Exception as exc:
                elapsed = time.time() - step_started_at
                print(
                    f"=== ERROR [{idx}/{len(runs)}] {algorithm.upper()} | {network} "
                    f"({elapsed:.1f}s): {exc} ==="
                )
                failures.append((algorithm, network, str(exc)))
                if args.stop_on_error:
                    break
            else:
                elapsed = time.time() - step_started_at
                print(
                    f"=== DONE  [{idx}/{len(runs)}] {algorithm.upper()} | {network} "
                    f"({elapsed:.1f}s) ==="
                )

    total_elapsed = time.time() - started_at
    print(f"\nCzas calkowity: {total_elapsed:.1f}s")

    if failures:
        print(f"Niepowodzenia: {len(failures)}")
        for algorithm, network, error in failures:
            print(f"- {algorithm.upper()} | {network}: {error}")
        raise SystemExit(1)

    print("Wszystkie treningi zakonczone sukcesem.")


if __name__ == "__main__":
    main()
