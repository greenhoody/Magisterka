#!/usr/bin/env python3
"""
Launch a single BotBowl game with the Flask UI so you can watch what the bots do.

The script bootstraps the same web server as ``botbowl/examples/server_example.py``,
but it also creates one specific game up front. Point your browser at the printed
URL and you will immediately see the live matchup.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import botbowl
from botbowl.ai.env import EnvConf
from botbowl.ai.registry import list_bots, make_bot
from botbowl.web import api, server

from a2c_env import a2c_scripted_actions
from ppo_residual_spatial_inception_agent import PPOAgent

# Import the example bots so they register themselves (scripted, random, etc.).
from botbowl.examples import scripted_bot_example  # noqa: F401
from botbowl.examples import random_bot_example  # noqa: F401


GAME_MODES_BY_SIZE = {
    11: "standard",
    7: "7v7",
    5: "5v5",
    3: "3v3",
    1: "1v1",
}


def register_ppo_bot(bot_id: str, model_path: Path, env_size: int, pathfinding: bool):
    """
    Register a PPOAgent under the requested bot ID so it shows up in the UI.
    """

    def factory(name: str):
        env_conf = EnvConf(size=env_size, pathfinding=pathfinding)
        return PPOAgent(
            name=name,
            env_conf=env_conf,
            scripted_func=a2c_scripted_actions,
            filename=str(model_path),
            exclude_pathfinding_moves=not pathfinding,
        )

    botbowl.register_bot(bot_id, factory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single BotBowl game with the web UI so you can observe your bot."
    )
    parser.add_argument(
        "--env-size",
        type=int,
        choices=sorted(GAME_MODES_BY_SIZE.keys()),
        default=7,
        help="Board size (affects both EnvConf and UI mode). Default: %(default)s",
    )
    parser.add_argument(
        "--pathfinding",
        action="store_true",
        help="Enable pathfinding when constructing EnvConf for custom PPO bots.",
    )
    parser.add_argument(
        "--home-bot",
        default="scripted",
        help="Registered bot ID for the home team. Default: %(default)s",
    )
    parser.add_argument(
        "--away-bot",
        default="random",
        help="Registered bot ID for the away team. Default: %(default)s",
    )
    parser.add_argument(
        "--home-team",
        default="human",
        help="Team template name for the home side. Default: %(default)s",
    )
    parser.add_argument(
        "--away-team",
        default="human",
        help="Team template name for the away side. Default: %(default)s",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "Optional PPO checkpoint. When provided the script registers a bot named "
            "'ppo-viewer' that loads the model and you can use it via --home-bot/--away-bot."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0" if os.path.exists("/.dockerenv") else "127.0.0.1",
        help="Host binding for the Flask UI. Default: %(default)s",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1234,
        help="Port for the Flask UI. Default: %(default)s",
    )
    return parser.parse_args()


def ensure_bot_exists(bot_id: str):
    if bot_id not in list_bots():
        raise SystemExit(
            f"Bot '{bot_id}' is not registered. Available bots: {', '.join(list_bots())}"
        )


def main():
    args = parse_args()

    mode = GAME_MODES_BY_SIZE[args.env_size]

    if args.model_path is not None:
        model_path = args.model_path.expanduser().resolve()
        if not model_path.is_file():
            raise SystemExit(f"PPO checkpoint '{model_path}' does not exist.")

        register_ppo_bot(
            bot_id="ppo-viewer",
            model_path=model_path,
            env_size=args.env_size,
            pathfinding=args.pathfinding,
        )
        if args.home_bot == "ppo-viewer" or args.away_bot == "ppo-viewer":
            print(f"Registered PPO bot at '{model_path}'.")

    ensure_bot_exists(args.home_bot)
    ensure_bot_exists(args.away_bot)

    home_agent = make_bot(args.home_bot)
    away_agent = make_bot(args.away_bot)

    game = api.new_game(
        home_team_name=args.home_team,
        away_team_name=args.away_team,
        home_agent=home_agent,
        away_agent=away_agent,
        game_mode=mode,
    )

    ui_url = f"http://{args.host}:{args.port}/#/game/{game.game_id}"
    print(f"Game ready: home={args.home_bot} vs away={args.away_bot}")
    print(f"Open {ui_url} in your browser to watch the match.")

    server.start_server(host=args.host, debug=True, use_reloader=False, port=args.port)


if __name__ == "__main__":
    main()
