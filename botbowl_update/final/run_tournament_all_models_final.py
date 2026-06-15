from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import inspect
import itertools
import json
import os
import random
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    import botbowl
    from botbowl.ai.env import BotBowlEnv, EnvConf

try:
    import botbowl as _botbowl

    BaseAgent = _botbowl.Agent
except ModuleNotFoundError:
    _botbowl = None

    class BaseAgent:
        def __init__(self, name: str):
            self.name = name


class LegacyPlayerAction:
    pass


def scripted_opening_action(game):
    from a2c_env import a2c_scripted_actions

    return a2c_scripted_actions(game)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Optional[Path]
    algo: str
    policy_module: str
    policy_class: str
    env_size: int
    source: str = "checkpoint"
    bot_candidates: tuple[str, ...] = field(default_factory=tuple)
    import_path: Optional[Path] = None


INTERNET_BOT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "drefsante": {
        "name": "Drefsante_AI_v.0.7",
        "relative_path": Path("INTERNET/Drefsante_AI-0.7/drefsante_bot.py"),
        "bot_candidates": ("Drefsante_AI_v.0.7",),
    },
    "grodbot": {
        "name": "GrodBot",
        "relative_path": Path("INTERNET/grodbot.py"),
        "bot_candidates": ("GrodBot",),
    },
    "minigrod": {
        "name": "minigrod",
        "relative_path": Path("INTERNET/minigrod/minigrod.py"),
        "bot_candidates": ("minigrod", "miniGrod"),
    },
}


def parse_algo(path: Path) -> str:
    name = path.name.lower()
    if "_a2c_" in name:
        return "a2c"
    if "_ppo_" in name:
        return "ppo"
    if "drefsante_curriculum" in name:
        return "curriculum"
    raise ValueError(f"Cannot infer algo from checkpoint name: {path}")


def parse_env_size(path: Path) -> Optional[int]:
    for part in path.parts:
        if part.startswith("botbowl-"):
            try:
                return int(part.split("-", 1)[1])
            except ValueError:
                return None
    return None


def discover_models(runs_dir: Path, include_pattern: str) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    pattern = f"*/botbowl-*/{include_pattern}"
    for ckpt_path in sorted(runs_dir.glob(pattern)):
        if not ckpt_path.is_file():
            continue
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format in {ckpt_path}")
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            raise ValueError(f"Checkpoint {ckpt_path} has no config dict")

        policy_module = cfg.get("policy_module")
        policy_class = cfg.get("policy_class", "CustomPolicy")
        if not policy_module:
            raise ValueError(f"Checkpoint {ckpt_path} has no policy_module in config")

        env_size = parse_env_size(ckpt_path)
        if env_size is None:
            raise ValueError(f"Cannot infer env size from path: {ckpt_path}")

        run_name = ckpt_path.parents[1].name
        model_name = f"{run_name}__{ckpt_path.stem}"
        specs.append(
            ModelSpec(
                name=model_name,
                path=ckpt_path.resolve(),
                algo=parse_algo(ckpt_path),
                policy_module=str(policy_module),
                policy_class=str(policy_class),
                env_size=env_size,
            )
        )
    return specs


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_random_bot_spec(env_size: int) -> ModelSpec:
    return ModelSpec(
        name="RandomBot",
        path=None,
        algo="bot",
        policy_module="",
        policy_class="",
        env_size=env_size,
        source="registered_bot",
        bot_candidates=("random", "RandomBot", "random_bot"),
    )


def filter_legacy_paths(game, paths):
    if paths is None:
        return paths
    if game is None:
        return paths

    pitch = getattr(getattr(game, "state", None), "pitch", None)
    width = getattr(pitch, "width", None)
    height = getattr(pitch, "height", None)
    if width is None or height is None:
        return paths

    valid_paths = []
    for path in paths:
        steps = getattr(path, "steps", None)
        if not steps:
            continue
        end = steps[-1]
        if 0 <= end.x < width and 0 <= end.y < height:
            valid_paths.append(path)
    return valid_paths


def install_ffai_compat_aliases(internet_dir: Optional[Path] = None) -> None:
    import botbowl

    sys.modules.setdefault("ffai", botbowl)
    if not hasattr(sys.modules["ffai"], "register_bot") and hasattr(botbowl, "register_bot"):
        setattr(sys.modules["ffai"], "register_bot", botbowl.register_bot)

    aliases = {
        "ffai.core": "botbowl.core",
        "ffai.core.model": "botbowl.core.model",
        "ffai.core.table": "botbowl.core.table",
        "ffai.core.procedure": "botbowl.core.procedure",
        "ffai.core.game": "botbowl.core.game",
        "ffai.ai": "botbowl.ai",
        "ffai.ai.proc_bot": "botbowl.ai.proc_bot",
        "ffai.ai.registry": "botbowl.ai.registry",
    }
    for old_name, new_name in aliases.items():
        if old_name in sys.modules:
            continue
        try:
            sys.modules[old_name] = importlib.import_module(new_name)
        except Exception:
            pass

    procedure_module = sys.modules.get("ffai.core.procedure")
    if procedure_module is not None and not hasattr(procedure_module, "PlayerAction"):
        setattr(procedure_module, "PlayerAction", LegacyPlayerAction)

    if "ffai.ai.pathfinding" not in sys.modules and internet_dir is not None:
        pathfinding_module = types.ModuleType("ffai.ai.pathfinding")
        minigrod_candidates = [
            internet_dir / "minigrod.py",
            internet_dir / "minigrod/minigrod.py",
            internet_dir.parent / "minigrod/minigrod.py",
        ]
        minigrod_path = next(
            (path.resolve() for path in minigrod_candidates if path.exists()),
            minigrod_candidates[-1].resolve(),
        )

        def load_minigrod_pathfinding():
            module_name = f"_ffai_compat_minigrod_{abs(hash(str(minigrod_path)))}"
            if module_name in sys.modules:
                return sys.modules[module_name]
            if not minigrod_path.exists():
                raise ImportError(f"Cannot provide ffai.ai.pathfinding: missing {minigrod_path}")

            spec = importlib.util.spec_from_file_location(module_name, minigrod_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module spec from {minigrod_path}")

            module = importlib.util.module_from_spec(spec)
            module.ffai = types.SimpleNamespace(register_bot=lambda *args, **kwargs: None)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def get_all_paths(*args, **kwargs):
            paths = load_minigrod_pathfinding().get_all_paths(*args, **kwargs)
            game = args[0] if args else kwargs.get("game")
            return filter_legacy_paths(game, paths)

        pathfinding_module.Path = object
        pathfinding_module.get_all_paths = get_all_paths
        sys.modules["ffai.ai.pathfinding"] = pathfinding_module
        setattr(sys.modules["ffai.ai"], "pathfinding", pathfinding_module)


def import_registered_bot_module(module_path: Path) -> None:
    module_path = module_path.resolve()
    install_ffai_compat_aliases(module_path.parent)
    module_name = f"internet_bot_{module_path.stem}_{abs(hash(str(module_path)))}"
    if module_name in sys.modules:
        patch_legacy_outcome_type_aliases(sys.modules[module_name])
        patch_legacy_pathfinding_bounds(sys.modules[module_name])
        patch_legacy_heat_map_bounds(sys.modules[module_name])
        return

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    module.ffai = sys.modules.get("ffai")
    sys.modules[module_name] = module

    old_cwd = Path.cwd()
    parent_str = str(module_path.parent)
    added_path = False
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        added_path = True

    try:
        os.chdir(module_path.parent)
        spec.loader.exec_module(module)
        patch_legacy_outcome_type_aliases(module)
        patch_legacy_pathfinding_bounds(module)
        patch_legacy_heat_map_bounds(module)
        if not hasattr(module, "PlayerAction"):
            module.PlayerAction = LegacyPlayerAction
    finally:
        os.chdir(old_cwd)
        if added_path:
            try:
                sys.path.remove(parent_str)
            except ValueError:
                pass


def patch_legacy_outcome_type_aliases(module: types.ModuleType) -> None:
    outcome_type = getattr(module, "OutcomeType", None)
    if outcome_type is None or hasattr(outcome_type, "BALL_DROPPED"):
        return

    for fallback_name in ("BALL_OUT_OF_BOUNDS", "BALL_HIT_GROUND", "FUMBLE", "TURNOVER"):
        fallback = getattr(outcome_type, fallback_name, None)
        if fallback is not None:
            setattr(outcome_type, "BALL_DROPPED", fallback)
            return


def patch_legacy_pathfinding_bounds(module: types.ModuleType) -> None:
    original_get_all_paths = getattr(module, "get_all_paths", None)
    if original_get_all_paths is None or getattr(original_get_all_paths, "_bounds_checked", False):
        return

    def get_all_paths(*args, **kwargs):
        paths = original_get_all_paths(*args, **kwargs)
        game = args[0] if args else kwargs.get("game")
        return filter_legacy_paths(game, paths)

    get_all_paths._bounds_checked = True
    module.get_all_paths = get_all_paths


def patch_legacy_heat_map_bounds(module: types.ModuleType) -> None:
    heat_map_cls = getattr(module, "FfHeatMap", None)
    if heat_map_cls is None or getattr(heat_map_cls, "_bounds_checked", False):
        return

    original_add_unit_paths = getattr(heat_map_cls, "add_unit_paths", None)
    if original_add_unit_paths is None:
        return

    def add_unit_paths(self, player, paths):
        width = len(getattr(self, "units_friendly", []))
        height = len(self.units_friendly[0]) if width > 0 else 0
        valid_paths = filter_legacy_paths(
            types.SimpleNamespace(
                state=types.SimpleNamespace(
                    pitch=types.SimpleNamespace(width=width, height=height)
                )
            ),
            paths,
        )
        return original_add_unit_paths(self, player, valid_paths)

    heat_map_cls.add_unit_paths = add_unit_paths
    heat_map_cls._bounds_checked = True


def install_legacy_game_api(game) -> None:
    if getattr(game, "_legacy_api_installed", False):
        return

    import botbowl
    from botbowl.core.model import ActionChoice

    if not hasattr(ActionChoice, "agi_rolls"):
        ActionChoice.agi_rolls = property(lambda self: self.rolls)

    original_safe_act = game._safe_act
    original_get_pass_distances_at = game.get_pass_distances_at

    def fill_legacy_skill_action_context(action):
        if action is None or getattr(action.action_type, "name", None) not in {
            "USE_SKILL",
            "DONT_USE_SKILL",
        }:
            return action

        choices = [
            choice
            for choice in game.state.available_actions
            if choice.action_type == action.action_type
        ]
        if len(choices) != 1:
            return action

        choice = choices[0]
        if getattr(action, "player", None) is None and len(choice.players) == 1:
            action.player = choice.players[0]
        if (
            getattr(action, "skill", None) is None
            and getattr(choice, "skill", None) is not None
        ):
            action.skill = choice.skill
        return action

    def action_choices(action_type):
        return [
            choice
            for choice in game.state.available_actions
            if choice.action_type == action_type
        ]

    def is_action_explicitly_available(action) -> bool:
        if action is None:
            return True
        for choice in action_choices(action.action_type):
            if choice.players and action.player not in choice.players:
                continue
            if choice.positions and action.position not in choice.positions:
                continue
            return True
        return False

    def normalize_legacy_select_player(action):
        if action is None or action.action_type != botbowl.ActionType.SELECT_PLAYER:
            return action

        choices = action_choices(botbowl.ActionType.SELECT_PLAYER)
        if not choices:
            return action

        if getattr(action, "player", None) is None and getattr(action, "position", None) is not None:
            player_at_position = game.get_player_at(action.position)
            if player_at_position is not None:
                for choice in choices:
                    if choice.players and not choice.positions and player_at_position in choice.players:
                        action.player = player_at_position
                        return action

        if is_action_explicitly_available(action):
            return action

        legal_players = []
        for choice in choices:
            for player in choice.players:
                if player not in legal_players:
                    legal_players.append(player)
        if len(legal_players) == 1:
            return botbowl.Action(botbowl.ActionType.SELECT_PLAYER, player=legal_players[0])
        return action

    def legacy_move_has_pending_handoff() -> bool:
        current_move = getattr(getattr(game, "actor", None), "current_move", None)
        action_steps = getattr(current_move, "action_steps", None)
        if not action_steps:
            return False
        return any(
            getattr(getattr(step, "action_type", None), "name", None) == "HANDOFF"
            for step in action_steps
        )

    def downgrade_legacy_handoff_start(action):
        if (
            action is None
            or getattr(action.action_type, "name", None) != "START_HANDOFF"
            or is_action_explicitly_available(action)
            or getattr(action, "player", None) is None
            or legacy_move_has_pending_handoff()
        ):
            return action

        move_action = botbowl.Action(botbowl.ActionType.START_MOVE, player=action.player)
        if is_action_explicitly_available(move_action):
            return move_action
        return action

    def is_high_kick_proc() -> bool:
        return type(game.state.stack.peek()).__name__ == "HighKick"

    def is_legacy_select_player_filter_proc() -> bool:
        return type(game.state.stack.peek()).__name__ in {"HighKick", "Touchback"}

    def select_player_allowed_players():
        for choice in game.state.available_actions:
            if choice.action_type == botbowl.ActionType.SELECT_PLAYER:
                return choice.team, choice.players
        return None, []

    def act_with_legacy_select_player_filter():
        if not is_legacy_select_player_filter_proc():
            return original_safe_act()

        allowed_team, allowed_players = select_player_allowed_players()
        if not allowed_players:
            return original_safe_act()

        original_get_players_on_pitch = game.get_players_on_pitch

        def get_players_on_pitch(*args, **kwargs):
            players = original_get_players_on_pitch(*args, **kwargs)
            team = kwargs.get("team", args[0] if args else None)
            if team == allowed_team:
                return [player for player in players if player in allowed_players]
            return players

        game.get_players_on_pitch = get_players_on_pitch
        try:
            return original_safe_act()
        finally:
            game.get_players_on_pitch = original_get_players_on_pitch

    def safe_act():
        action = act_with_legacy_select_player_filter()
        action = fill_legacy_skill_action_context(action)
        action = downgrade_legacy_handoff_start(action)
        action = normalize_legacy_select_player(action)
        if (
            action is not None
            and is_high_kick_proc()
            and getattr(action.action_type, "name", None) == "PLACE_PLAYER"
        ):
            select_player = botbowl.Action(
                botbowl.ActionType.SELECT_PLAYER,
                player=action.player,
            )
            if is_action_explicitly_available(select_player):
                return select_player
            select_none = botbowl.Action(botbowl.ActionType.SELECT_NONE)
            if is_action_explicitly_available(select_none):
                return select_none
        return action

    def get_pass_distances_at(*args, **kwargs):
        if len(args) == 2 and hasattr(args[1], "x") and hasattr(args[1], "y"):
            passer, position = args
            return original_get_pass_distances_at(
                passer,
                game.get_ball(),
                position,
                **kwargs,
            )
        if len(args) == 3 and isinstance(args[2], bool):
            passer, position, dump_off = args
            return original_get_pass_distances_at(
                passer,
                game.get_ball(),
                position,
                dump_off=dump_off,
                **kwargs,
            )
        return original_get_pass_distances_at(*args, **kwargs)

    game._safe_act = safe_act
    game.get_pass_distances_at = get_pass_distances_at
    game._legacy_api_installed = True


def install_action_debug_logging(game) -> None:
    if getattr(game, "_action_debug_installed", False):
        return

    original_safe_act = game._safe_act

    def safe_act():
        action = original_safe_act()
        if action is not None and not game._is_action_allowed(action):
            log_illegal_action_context(game, action)
        return action

    game._safe_act = safe_act
    game._action_debug_installed = True


def team_debug_label(game, team) -> str:
    if team is None:
        return "None"
    side = "home" if team == game.state.home_team else "away"
    return f"{getattr(team, 'name', '?')}[{side}]"


def square_debug_label(game, square) -> str:
    if square is None:
        return "None"
    tile = game.arena.board[square.y][square.x].name
    home_side = game.is_team_side(square, game.state.home_team)
    away_side = game.is_team_side(square, game.state.away_team)
    return (
        f"({square.x},{square.y}) tile={tile} "
        f"home_side={home_side} away_side={away_side}"
    )


def summarize_positions(game, positions, wanted_position) -> str:
    if not positions:
        return "positions=0"

    xs = [pos.x for pos in positions if pos is not None]
    ys = [pos.y for pos in positions if pos is not None]
    summary = [f"positions={len(positions)}"]
    if xs and ys:
        summary.append(f"x=[{min(xs)}..{max(xs)}] y=[{min(ys)}..{max(ys)}]")

    if wanted_position is not None:
        closest = sorted(
            (pos for pos in positions if pos is not None),
            key=lambda pos: abs(pos.x - wanted_position.x) + abs(pos.y - wanted_position.y),
        )[:5]
        summary.append(
            "closest="
            + ", ".join(f"({pos.x},{pos.y})" for pos in closest)
        )
    return " ".join(summary)


def log_illegal_action_context(game, action) -> None:
    proc = game.state.stack.peek()
    actor = game.actor
    active_team = game.active_team
    print("=== DEBUG_ACTION_NOT_ALLOWED ===")
    print(f"game_id={game.game_id} half={game.state.half} round={game.state.round}")
    print(f"procedure={type(proc).__name__}")
    print(f"actor={getattr(actor, 'name', None)} active_team={team_debug_label(game, active_team)}")
    print(f"action={action.to_json() if hasattr(action, 'to_json') else action}")

    player = getattr(action, "player", None)
    if player is not None:
        print(
            "action_player="
            f"{getattr(player, 'name', '?')} id={getattr(player, 'player_id', '?')} "
            f"team={team_debug_label(game, getattr(player, 'team', None))} "
            f"position={square_debug_label(game, getattr(player, 'position', None))}"
        )
    print(f"action_position={square_debug_label(game, getattr(action, 'position', None))}")

    current_move = getattr(actor, "current_move", None)
    if current_move is not None:
        steps = getattr(current_move, "action_steps", [])
        print(
            "actor_current_move="
            f"description={getattr(current_move, 'description', None)!r} "
            f"remaining_steps={len(steps)}"
        )

    for idx, choice in enumerate(game.state.available_actions):
        if choice.action_type != action.action_type:
            continue
        print(
            f"choice[{idx}] team={team_debug_label(game, choice.team)} "
            f"players={len(choice.players)} "
            f"player_allowed={player in choice.players if player is not None else None} "
            f"{summarize_positions(game, choice.positions, getattr(action, 'position', None))} "
            f"position_allowed={getattr(action, 'position', None) in choice.positions}"
        )
    print("=== END_DEBUG_ACTION_NOT_ALLOWED ===")


def discover_internet_bots(
    base_dir: Path,
    env_size: int,
    selected: Optional[Sequence[str]] = None,
) -> tuple[List[ModelSpec], List[str]]:
    selected_keys = list(selected) if selected is not None else list(INTERNET_BOT_DEFINITIONS)
    specs: List[ModelSpec] = []
    warnings: List[str] = []

    for key in selected_keys:
        definition = INTERNET_BOT_DEFINITIONS.get(key)
        if definition is None:
            warnings.append(f"Unknown INTERNET bot key: {key}")
            continue

        import_path = (base_dir / definition["relative_path"]).resolve()
        if not import_path.exists():
            warnings.append(f"INTERNET bot file missing: {import_path}")
            continue

        try:
            import_registered_bot_module(import_path)
        except Exception as exc:
            warnings.append(f"Skipping {definition['name']} ({import_path}): {exc}")
            continue

        specs.append(
            ModelSpec(
                name=definition["name"],
                path=import_path,
                algo="bot",
                policy_module="",
                policy_class="",
                env_size=env_size,
                source="imported_registered_bot",
                bot_candidates=tuple(definition["bot_candidates"]),
                import_path=import_path,
            )
        )

    return specs, warnings


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_game_assets(env_conf, fast_mode: bool):
    import botbowl

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


def build_policy_from_checkpoint(
    spec: ModelSpec,
    spatial_shape: tuple[int, int, int],
    non_spatial_size: int,
    action_space: int,
) -> torch.nn.Module:
    if spec.path is None:
        raise ValueError(f"Spec {spec.name} does not point to a checkpoint file.")

    checkpoint = torch.load(spec.path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, torch.nn.Module):
        policy = checkpoint
        policy.eval()
        return policy

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format in {spec.path}")

    state_dict = checkpoint["model"]
    module = importlib.import_module(spec.policy_module)
    policy_cls = getattr(module, spec.policy_class)
    signature = inspect.signature(policy_cls.__init__)

    base_kwargs: Dict[str, Any] = {
        "spatial_shape": spatial_shape,
        "non_spatial_size": non_spatial_size,
        "action_space": action_space,
    }

    hidden_candidates: List[Optional[int]] = [None]
    if "hidden_nodes" in signature.parameters:
        inferred_hidden: List[int] = []
        for key in ("trunk.0.weight", "actor.0.weight", "critic.0.weight"):
            tensor = state_dict.get(key)
            if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 1:
                inferred_hidden.append(int(tensor.shape[0]))

        config_hidden = checkpoint.get("config", {}).get("hidden_nodes")
        ordered_candidates: List[Optional[int]] = []
        for candidate in [config_hidden, *inferred_hidden, action_space, 128, 256, 512, None]:
            if candidate not in ordered_candidates:
                ordered_candidates.append(candidate)
        hidden_candidates = ordered_candidates

    last_error: Optional[Exception] = None
    for hidden in hidden_candidates:
        kwargs = dict(base_kwargs)
        if hidden is not None:
            kwargs["hidden_nodes"] = hidden
        try:
            policy = policy_cls(**kwargs)
            policy.load_state_dict(state_dict, strict=True)
            policy.eval()
            return policy
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to instantiate/load policy for {spec.name} ({spec.path}): {last_error}"
    )


class CheckpointAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        env_conf,
        spec: ModelSpec,
        scripted_opening: bool,
    ):
        import botbowl
        from botbowl.ai.env import BotBowlEnv

        super().__init__(name)
        self.env = BotBowlEnv(env_conf)
        self.spec = spec
        self.scripted_opening = scripted_opening
        self.action_queue: List[Any] = []

        spatial_obs, non_spatial_obs, action_mask = self.env.reset()
        self.policy = build_policy_from_checkpoint(
            spec=spec,
            spatial_shape=tuple(spatial_obs.shape),
            non_spatial_size=int(non_spatial_obs.shape[0]),
            action_space=int(action_mask.shape[0]),
        )

    def new_game(self, game, team):
        return None

    @staticmethod
    def _update_obs(array: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.from_numpy(array.copy()), dim=0)

    def act(self, game):
        if self.action_queue:
            return self.action_queue.pop(0)

        if self.scripted_opening:
            scripted_action = scripted_opening_action(game)
            if scripted_action is not None:
                return scripted_action

        self.env.game = game
        spatial_obs, non_spatial_obs, action_mask = map(
            CheckpointAgent._update_obs, self.env.get_state()
        )
        non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=0)

        with torch.no_grad():
            _, actions = self.policy.act(
                spatial_obs.float(),
                non_spatial_obs.float(),
                action_mask,
            )

        action_idx = int(actions[0])
        action_objects = self.env._compute_action(action_idx)
        if not action_objects:
            raise RuntimeError(f"Policy action {action_idx} produced no BotBowl actions.")
        self.action_queue = action_objects
        return self.action_queue.pop(0)

    def end_game(self, game):
        return None


def build_registered_bot(spec: ModelSpec):
    import botbowl

    last_error: Optional[Exception] = None
    for candidate in spec.bot_candidates:
        try:
            agent = botbowl.make_bot(candidate)
            agent.name = spec.name
            return agent
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to instantiate registered bot for {spec.name}. "
        f"Tried: {', '.join(spec.bot_candidates)}. Last error: {last_error}"
    )


def build_agent(spec: ModelSpec, env_conf, scripted_opening: bool):
    if spec.source == "checkpoint":
        return CheckpointAgent(
            name=spec.name,
            env_conf=env_conf,
            spec=spec,
            scripted_opening=scripted_opening,
        )
    if spec.source == "registered_bot":
        return build_registered_bot(spec)
    if spec.source == "imported_registered_bot":
        if spec.import_path is None:
            raise ValueError(f"Imported bot {spec.name} has no import_path")
        import_registered_bot_module(spec.import_path)
        return build_registered_bot(spec)
    raise ValueError(f"Unsupported model source: {spec.source}")


def play_single_game(
    home: ModelSpec,
    away: ModelSpec,
    env_conf,
    fast_mode: bool,
    game_id: int,
    seed: Optional[int],
    scripted_opening: bool,
    debug_actions: bool,
) -> Dict[str, Any]:
    import botbowl

    config, ruleset, arena, home_team, away_team = load_game_assets(env_conf, fast_mode)
    home_agent = build_agent(home, env_conf, scripted_opening)
    away_agent = build_agent(away, env_conf, scripted_opening)

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
    install_legacy_game_api(game)
    if debug_actions:
        install_action_debug_logging(game)
    seed_everything(seed)
    if seed is not None:
        game_rng = getattr(game, "rng", None) or getattr(game, "rnd", None)
        if game_rng is not None:
            game_rng.seed(seed)
    game.init()

    home_team_state = game.get_agent_team(home_agent).state
    away_team_state = game.get_agent_team(away_agent).state
    winner_agent = game.get_winner()
    if winner_agent is None:
        winner = None
    elif winner_agent == home_agent:
        winner = home.name
    else:
        winner = away.name

    home_agent.end_game(game)
    away_agent.end_game(game)
    return {
        "game_id": game_id,
        "home": home.name,
        "away": away.name,
        "home_score": int(home_team_state.score),
        "away_score": int(away_team_state.score),
        "winner": winner,
        "seed": seed,
        "rounds": int(game.state.round),
        "home_turns": int(home_team_state.turn),
        "away_turns": int(away_team_state.turn),
    }


def run_tournament(
    models: List[ModelSpec],
    env_conf,
    games_per_pair: int,
    fast_mode: bool,
    seed: Optional[int],
    scripted_opening: bool,
    debug_actions: bool,
):
    rng = random.Random(seed)
    series_summaries = []
    game_logs = []
    total_games = (len(models) * (len(models) - 1) // 2) * games_per_pair
    games_done = 0
    started_at = time.time()

    for pair_index, (left, right) in enumerate(
        itertools.combinations(models, 2), start=1
    ):
        print(
            f"Pair {pair_index}: {left.name} vs {right.name} "
            f"({games_per_pair} game(s))"
        )
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
                scripted_opening=scripted_opening,
                debug_actions=debug_actions,
            )
            summary["games"] += 1
            summary["touchdowns"][result["home"]] += result["home_score"]
            summary["touchdowns"][result["away"]] += result["away_score"]
            if result["winner"] is None:
                summary["draws"] += 1
            else:
                summary["wins"][result["winner"]] += 1
            game_logs.append(result)
            games_done += 1
            elapsed = time.time() - started_at
            avg_game_time = elapsed / games_done if games_done > 0 else 0.0
            eta = avg_game_time * max(total_games - games_done, 0)
            print(
                f"  Game {games_done}/{total_games}: {result['home']} {result['home_score']}"
                f" - {result['away_score']} {result['away']} | "
                f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}"
            )

        series_summaries.append(summary)

    return series_summaries, game_logs


def build_global_table(models: List[ModelSpec], series: List[Dict[str, Any]]):
    table: Dict[str, Dict[str, Any]] = {
        m.name: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "games": 0,
            "touchdowns_for": 0,
            "touchdowns_against": 0,
            "algo": m.algo,
            "path": str(m.path),
        }
        for m in models
    }

    for s in series:
        left, right = s["pair"]
        l_wins = s["wins"][left]
        r_wins = s["wins"][right]
        draws = s["draws"]
        games = s["games"]

        table[left]["wins"] += l_wins
        table[left]["draws"] += draws
        table[left]["losses"] += max(0, games - l_wins - draws)
        table[left]["games"] += games
        table[left]["touchdowns_for"] += s["touchdowns"][left]
        table[left]["touchdowns_against"] += s["touchdowns"][right]

        table[right]["wins"] += r_wins
        table[right]["draws"] += draws
        table[right]["losses"] += max(0, games - r_wins - draws)
        table[right]["games"] += games
        table[right]["touchdowns_for"] += s["touchdowns"][right]
        table[right]["touchdowns_against"] += s["touchdowns"][left]

    ranking = sorted(
        (
            {
                "name": name,
                **stats,
                "points": stats["wins"] * 3 + stats["draws"],
            }
            for name, stats in table.items()
        ),
        key=lambda x: (
            x["points"],
            x["wins"],
            x["touchdowns_for"] - x["touchdowns_against"],
            x["touchdowns_for"],
        ),
        reverse=True,
    )
    return ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tournament for all checkpoints from final/runs using "
            "saved policy_module/policy_class and state_dict checkpoints."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory with run folders. Default: %(default)s",
    )
    parser.add_argument(
        "--include-pattern",
        default="*_final.pt",
        help="Checkpoint filename pattern under run dirs. Default: %(default)s",
    )
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=2,
        help="How many games per model pair. Default: %(default)s",
    )
    parser.add_argument(
        "--env-size",
        type=int,
        default=None,
        help="Force environment size. By default inferred from checkpoint paths.",
    )
    parser.add_argument(
        "--trained-env-size",
        type=int,
        default=None,
        help=(
            "Only include models trained on the given environment size "
            "(derived from botbowl-<size> in checkpoint paths)."
        ),
    )
    parser.add_argument(
        "--pathfinding",
        action="store_true",
        help="Enable pathfinding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=True,
        help="Run games in fast mode (default).",
    )
    parser.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="Disable fast mode.",
    )
    parser.add_argument(
        "--no-scripted-opening",
        dest="scripted_opening",
        action="store_false",
        default=True,
        help="Disable scripted opening actions.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional cap on number of discovered models.",
    )
    parser.add_argument(
        "--include-random-bot",
        action="store_true",
        help="Include BotBowl registered RandomBot in the tournament.",
    )
    parser.add_argument(
        "--include-internet-bots",
        action="store_true",
        help="Include bots defined in the INTERNET folder when their imports succeed.",
    )
    parser.add_argument(
        "--internet-bots",
        nargs="+",
        choices=sorted(INTERNET_BOT_DEFINITIONS.keys()),
        default=None,
        help=(
            "Subset of INTERNET bots to load. "
            f"Available: {', '.join(sorted(INTERNET_BOT_DEFINITIONS.keys()))}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tournament_all_models_final.json"),
        help="Output JSON report path. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered models and exit.",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="Print detailed context before BotBowl rejects an illegal action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models = discover_models(args.runs_dir.resolve(), args.include_pattern)
    if args.trained_env_size is not None:
        models = [m for m in models if m.env_size == args.trained_env_size]
    if args.max_models is not None:
        models = models[: args.max_models]

    env_sizes = sorted(set(m.env_size for m in models))
    if args.env_size is not None:
        env_size = args.env_size
    else:
        if len(env_sizes) != 1:
            raise SystemExit(
                "Multiple env sizes detected: "
                + ", ".join(map(str, env_sizes))
                + ". Use --env-size."
            )
        env_size = env_sizes[0]

    if args.include_random_bot:
        models.append(make_random_bot_spec(env_size))

    if args.include_internet_bots:
        internet_models, internet_warnings = discover_internet_bots(
            base_dir=Path(__file__).resolve().parent,
            env_size=env_size,
            selected=args.internet_bots,
        )
        models.extend(internet_models)
        for warning in internet_warnings:
            print(f"WARNING: {warning}")

    if len(models) < 2:
        raise SystemExit("Need at least two tournament participants after filtering.")

    print(f"Discovered models: {len(models)}")
    for idx, model in enumerate(models, start=1):
        path_display = str(model.path) if model.path is not None else model.source
        print(f"[{idx}] {model.name} | algo={model.algo} | path={path_display}")

    if args.dry_run:
        print("Dry-run finished.")
        return

    from botbowl.ai.env import EnvConf

    env_conf = EnvConf(size=env_size, pathfinding=args.pathfinding)
    series, games = run_tournament(
        models=models,
        env_conf=env_conf,
        games_per_pair=args.games_per_pair,
        fast_mode=args.fast_mode,
        seed=args.seed,
        scripted_opening=args.scripted_opening,
        debug_actions=args.debug_actions,
    )

    ranking = build_global_table(models, series)
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env_size": env_size,
        "pathfinding": args.pathfinding,
        "debug_actions": args.debug_actions,
        "games_per_pair": args.games_per_pair,
        "models_count": len(models),
        "models": [
            {
                "name": m.name,
                "path": str(m.path) if m.path is not None else None,
                "algo": m.algo,
                "policy_module": m.policy_module,
                "policy_class": m.policy_class,
                "env_size": m.env_size,
                "source": m.source,
            }
            for m in models
        ],
        "ranking": ranking,
        "series": series,
        "games": games,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved tournament report to {args.output.resolve()}")


if __name__ == "__main__":
    main()
