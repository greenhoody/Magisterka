from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import botbowl
from botbowl.ai.env import EnvConf

from a2c_env import a2c_scripted_actions
from a2c_residual_spatial_inception_agent import A2CAgent
from ppo_residual_spatial_inception_agent import PPOAgent


SUPPORTED_ALGOS = {"a2c", "ppo"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    algo: str

    def instantiate(self, env_conf: EnvConf):
        if self.algo == "a2c":
            return A2CAgent(
                name=self.name,
                env_conf=env_conf,
                scripted_func=a2c_scripted_actions,
                filename=str(self.path),
            )
        if self.algo == "ppo":
            return PPOAgent(
                name=self.name,
                env_conf=env_conf,
                scripted_func=a2c_scripted_actions,
                filename=str(self.path),
            )
        raise ValueError(f"Unsupported algo '{self.algo}'")


def parse_model_spec(raw: str, base_dir: Path) -> ModelSpec:
    """
    Expected format: NAME:RELATIVE_OR_ABS_PATH:ALGO
    Example: champ:botbowl-3/735d9750.nn:a2c
    """
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Model spec '{raw}' must look like NAME:PATH:ALGO"
        )
    name, rel_path, algo = parts
    algo = algo.lower()
    if algo not in SUPPORTED_ALGOS:
        raise argparse.ArgumentTypeError(
            f"Model '{name}' uses unsupported algo '{algo}'. "
            f"Allowed: {', '.join(sorted(SUPPORTED_ALGOS))}."
        )
    candidate = Path(rel_path)
    if not candidate.is_file():
        candidate = (base_dir / rel_path).resolve()
    if not candidate.is_file():
        raise argparse.ArgumentTypeError(
            f"Model path '{rel_path}' (resolved to '{candidate}') does not exist."
        )
    return ModelSpec(name=name, path=candidate, algo=algo)


def seed_everything(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_game_assets(env_conf: EnvConf, fast_mode: bool):
    config = botbowl.load_config("bot-bowl")
    config.competition_mode = False
    config.pathfinding_enabled = env_conf.pathfinding
    config.fast_mode = fast_mode
    config.debug_mode = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    home_team = botbowl.load_team_by_filename("human", ruleset)
    away_team = botbowl.load_team_by_filename("human", ruleset)
    return config, ruleset, arena, home_team, away_team


def play_single_game(
    home: ModelSpec,
    away: ModelSpec,
    env_conf: EnvConf,
    fast_mode: bool,
    game_id: int,
    seed: Optional[int],
) -> Dict[str, Optional[str]]:
    config, ruleset, arena, home_team, away_team = load_game_assets(env_conf, fast_mode)
    home_agent = home.instantiate(env_conf)
    away_agent = away.instantiate(env_conf)
    game = botbowl.Game(
        game_id,
        home_team,
        away_team,
        home_agent,
        away_agent,
        config,
        arena=arena,
        ruleset=ruleset,
    )
    seed_everything(seed)
    if seed is not None:
        game.rnd.seed(seed)
    game.init()
    home_team_state = game.get_agent_team(home_agent).state
    away_team_state = game.get_agent_team(away_agent).state
    winner_agent = game.get_winner()
    if winner_agent is None:
        winner_name: Optional[str] = None
    elif winner_agent == home_agent:
        winner_name = home.name
    else:
        winner_name = away.name
    home_agent.end_game(game)
    away_agent.end_game(game)
    return {
        "game_id": game.game_id,
        "home": home.name,
        "away": away.name,
        "home_score": home_team_state.score,
        "away_score": away_team_state.score,
        "winner": winner_name,
        "seed": seed,
    }


def run_tournament(
    models: List[ModelSpec],
    env_conf: EnvConf,
    games_per_pair: int,
    fast_mode: bool,
    seed: Optional[int],
):
    rng = random.Random(seed)
    series_summaries = []
    game_logs = []
    for pair_index, (left, right) in enumerate(
        itertools.combinations(models, 2), start=1
    ):
        summary = {
            "pair": [left.name, right.name],
            "games": 0,
            "wins": {left.name: 0, right.name: 0},
            "draws": 0,
            "touchdowns": {left.name: 0, right.name: 0},
        }
        for local_idx in range(games_per_pair):
            home, away = (left, right) if local_idx % 2 == 0 else (right, left)
            game_seed = rng.randint(0, 2**31 - 1) if seed is not None else None
            result = play_single_game(
                home=home,
                away=away,
                env_conf=env_conf,
                fast_mode=fast_mode,
                game_id=(pair_index * 1000) + local_idx,
                seed=game_seed,
            )
            summary["games"] += 1
            summary["touchdowns"][result["home"]] += result["home_score"]
            summary["touchdowns"][result["away"]] += result["away_score"]
            if result["winner"] is None:
                summary["draws"] += 1
            else:
                summary["wins"][result["winner"]] += 1
            game_logs.append(result)
        series_summaries.append(summary)
    return series_summaries, game_logs


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a round-robin tournament between saved BotBowl models. "
            "Each --model argument must look like NAME:PATH:ALGO where "
            "ALGO ∈ {a2c, ppo}. PATH can be absolute or relative to --models-dir."
        )
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Model spec in the form NAME:PATH:ALGO (repeatable).",
    )
    parser.add_argument(
        "--models-dir",
        default="../models",
        help="Base directory used when a PATH is relative. Default: %(default)s",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        default=3,
        help="Environment size passed to EnvConf. Default: %(default)s",
    )
    parser.add_argument(
        "--pathfinding",
        action="store_true",
        help="Enable pathfinding in EnvConf and BotBowl config.",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=4,
        help="How many games to run for each pairing. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional base seed for reproducible tournaments.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=True,
        help="Run BotBowl games in fast mode (default: %(default)s).",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable BotBowl fast mode.",
    )
    parser.add_argument(
        "--output",
        default="tournament_results.json",
        help="Path to the JSON report that will be written. Default: %(default)s",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(args.models_dir).resolve()
    model_specs = [
        parse_model_spec(raw, base_dir=base_dir) for raw in args.models
    ]
    if len(model_specs) < 2:
        parser.error("At least two --model specs must be provided.")

    env_conf = EnvConf(size=args.env_size, pathfinding=args.pathfinding)
    series, games = run_tournament(
        models=model_specs,
        env_conf=env_conf,
        games_per_pair=args.games_per_pair,
        fast_mode=args.fast_mode,
        seed=args.seed,
    )

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env_size": args.env_size,
        "pathfinding": args.pathfinding,
        "games_per_pair": args.games_per_pair,
        "models": [
            {"name": spec.name, "path": str(spec.path), "algo": spec.algo}
            for spec in model_specs
        ],
        "series": series,
        "games": games,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved tournament report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
