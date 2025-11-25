#!/usr/bin/env python3
"""
Play a saved BotBowl model against the built-in random bot.

Example:
    python play_vs_random.py \
        --model-name champ \
        --model-path models/botbowl-7/foo.nn \
        --algo ppo \
        --env-size 7 \
        --games 10 \
        --output results/random_match.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def summarize_last_report(game: botbowl.Game) -> Optional[Dict[str, Any]]:
    if not game.state.reports:
        return None
    report = game.state.reports[-1]
    outcome = getattr(report, "outcome_type", None)
    team = getattr(report, "team", None)
    return {
        "outcome": getattr(outcome, "name", str(outcome)),
        "team": getattr(team, "name", None),
    }


def load_game_assets(env_conf: EnvConf, fast_mode: bool):
    config = copy.deepcopy(env_conf.config)
    config.competition_mode = False
    config.pathfinding_enabled = getattr(env_conf, "pathfinding", False)
    config.fast_mode = fast_mode
    config.debug_mode = False
    ruleset = botbowl.load_rule_set(config.ruleset)
    arena = botbowl.load_arena(config.arena)
    board_size = getattr(env_conf, "size", 11)
    home_team = botbowl.load_team_by_filename("human", ruleset, board_size=board_size)
    away_team = botbowl.load_team_by_filename("human", ruleset, board_size=board_size)
    return config, ruleset, arena, home_team, away_team


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def play_single_game(
    model: ModelSpec,
    env_conf: EnvConf,
    fast_mode: bool,
    game_id: int,
    model_is_home: bool,
    seed: Optional[int],
):
    config, ruleset, arena, home_team, away_team = load_game_assets(
        env_conf, fast_mode
    )
    scripted = a2c_scripted_actions
    model_agent = model.instantiate(env_conf)
    random_agent = botbowl.make_bot("random")

    if model_is_home:
        home_agent, away_agent = model_agent, random_agent
    else:
        home_agent, away_agent = random_agent, model_agent

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
    home_state = game.get_agent_team(home_agent).state
    away_state = game.get_agent_team(away_agent).state
    winner = game.get_winner()
    if winner is None:
        winner_name = None
    elif winner == home_agent:
        winner_name = model.name if model_is_home else "random"
    else:
        winner_name = "random" if model_is_home else model.name

    model_agent.end_game(game)
    random_agent.end_game(game)

    return {
        "game_id": game.game_id,
        "model_home": model_is_home,
        "home_score": home_state.score,
        "away_score": away_state.score,
        "winner": winner_name,
        "rounds": game.state.round,
        "home_turns": home_state.turn,
        "away_turns": away_state.turn,
        "last_report": summarize_last_report(game),
    }


def run_series(
    model: ModelSpec,
    env_conf: EnvConf,
    games: int,
    fast_mode: bool,
    alternating_home: bool,
    seed: Optional[int],
):
    rng = random.Random(seed)
    summary = {
        "model": model.name,
        "games": 0,
        "wins": {model.name: 0, "random": 0},
        "draws": 0,
        "touchdowns": {model.name: 0, "random": 0},
    }
    logs: List[Dict[str, Any]] = []
    for idx in range(games):
        model_home = idx % 2 == 0 if alternating_home else True
        game_seed = rng.randint(0, 2**31 - 1) if seed is not None else None
        result = play_single_game(
            model=model,
            env_conf=env_conf,
            fast_mode=fast_mode,
            game_id=idx,
            model_is_home=model_home,
            seed=game_seed,
        )
        summary["games"] += 1
        model_td = (
            result["home_score"] if model_home else result["away_score"]
        )
        random_td = (
            result["away_score"] if model_home else result["home_score"]
        )
        summary["touchdowns"][model.name] += model_td
        summary["touchdowns"]["random"] += random_td
        if result["winner"] is None:
            summary["draws"] += 1
        else:
            summary["wins"][result["winner"]] += 1
        logs.append(result)
    return summary, logs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pit a saved BotBowl model against the built-in random bot."
    )
    parser.add_argument("--model-name", required=True, help="Display name for the model.")
    parser.add_argument("--model-path", required=True, help="Path to the saved model file.")
    parser.add_argument(
        "--algo",
        required=True,
        choices=sorted(SUPPORTED_ALGOS),
        help="Algorithm used to train the model (affects loader).",
    )
    parser.add_argument("--env-size", type=int, default=7, help="BotBowl gym environment size.")
    parser.add_argument(
        "--games",
        type=int,
        default=4,
        help="How many matches to run (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducibility.",
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
        help="Disable fast mode rendering.",
    )
    parser.add_argument(
        "--alternate-home",
        action="store_true",
        help="Alternate which side (home/away) the model plays.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file to store the match summary/logs.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.is_file():
        parser.error(f"Model path '{model_path}' does not exist.")
    spec = ModelSpec(name=args.model_name, path=model_path, algo=args.algo.lower())

    env_conf = EnvConf(size=args.env_size)
    summary, games = run_series(
        model=spec,
        env_conf=env_conf,
        games=args.games,
        fast_mode=args.fast_mode,
        alternating_home=args.alternate_home,
        seed=args.seed,
    )

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": {"name": spec.name, "path": str(spec.path), "algo": spec.algo},
        "env_size": args.env_size,
        "fast_mode": args.fast_mode,
        "games": args.games,
        "alternate_home": args.alternate_home,
        "summary": summary,
        "match_logs": games,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Saved report to {out_path.resolve()}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
