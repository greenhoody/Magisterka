from __future__ import annotations

import csv
import gc
import importlib
import importlib.util
import inspect
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_DREFSANTE_DIR = Path(__file__).resolve().parent / "INTERNET/Drefsante_AI-0.7"
_DREFSANTE_JAR = _DREFSANTE_DIR / "bloodbowl-0.7.jar"
import botbowl

from a2c_env import A2C_Reward
from training_env import resolve_env_size
from training_run_naming import build_unique_run_paths


DREFSANTE_MODULE_PATH = Path("INTERNET/Drefsante_AI-0.7/drefsante_bot.py")
DREFSANTE_BOT_NAME = "Drefsante_AI_v.0.7"


def configure_drefsante_classpath() -> None:
    if not _DREFSANTE_JAR.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku JAR Drefsante: {_DREFSANTE_JAR}")

    jar_classpath = str(_DREFSANTE_JAR)
    dir_classpath = str(_DREFSANTE_DIR)
    current_classpath = os.environ.get("CLASSPATH", "")
    classpath_entries = [entry for entry in current_classpath.split(os.pathsep) if entry]
    for entry in (jar_classpath, dir_classpath):
        if entry not in classpath_entries:
            classpath_entries.insert(0, entry)
    os.environ["CLASSPATH"] = os.pathsep.join(classpath_entries)

    import jnius_config

    configured = getattr(jnius_config, "classpath", None)
    if configured is None:
        jnius_config.set_classpath(dir_classpath, jar_classpath)
    else:
        missing = [
            entry for entry in (dir_classpath, jar_classpath) if entry not in configured
        ]
        if missing:
            jnius_config.add_classpath(*missing)


def verify_drefsante_java_class() -> None:
    configure_drefsante_classpath()
    from jnius import autoclass

    try:
        autoclass("be.drefsante.bloodbowl.presenter.FFAIProxy")
    except Exception as exc:
        if "class file version 64.0" in str(exc) or "UnsupportedClassVersionError" in str(exc):
            raise RuntimeError(
                "Drefsante bloodbowl-0.7.jar wymaga Javy 20 lub nowszej. "
                "Aktualna JVM jest za stara; Java 17 obsluguje maksymalnie "
                "class file version 61.0, a ten JAR ma 64.0."
            ) from exc
        raise


def allow_redundant_jnius_classpath_updates() -> None:
    try:
        import jnius_config
    except Exception:
        return

    if getattr(jnius_config, "_drefsante_safe_set_classpath", False):
        return

    original_set_classpath = jnius_config.set_classpath

    def safe_set_classpath(*path):
        try:
            return original_set_classpath(*path)
        except ValueError as exc:
            if "VM is already running" in str(exc):
                return None
            raise

    jnius_config.set_classpath = safe_set_classpath
    jnius_config._drefsante_safe_set_classpath = True


def cleanup_java_ui_memory() -> None:
    try:
        from jnius import autoclass

        Window = autoclass("java.awt.Window")
        for window in Window.getWindows():
            try:
                window.dispose()
            except Exception:
                pass

        System = autoclass("java.lang.System")
        Runtime = autoclass("java.lang.Runtime")
        System.gc()
        Runtime.getRuntime().gc()
    except Exception:
        pass


def cleanup_drefsante_agent(agent) -> None:
    try:
        setattr(agent, "ffaiProxy", None)
    except Exception:
        pass
    cleanup_java_ui_memory()


@dataclass
class Config:
    seed: int = random.randint(0, 10000)
    env_size_default: int = 11
    policy_module: str = "a2c_manifold_dynamic_hyperconnection_11x11"
    policy_class: str = "CustomPolicy"
    out_dir_root: str = "runs"

    imitation_hours: float = 24.0
    reinforcement_hours: float = 48.0
    imitation_lr: float = 1e-4
    reinforcement_lr: float = 3e-5
    imitation_batch_size: int = 32
    imitation_train_batches_per_cycle: int = 4
    imitation_recent_buffer_size: int = 20_000
    imitation_archive_buffer_size: int = 0
    imitation_archive_sample_fraction: float = 0.0

    rollout_len: int = 32
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_interval: int = 10
    checkpoint_interval_minutes: float = 30.0
    cpu_threads: int = 0
    cpu_interop_threads: int = 0
    java_headless: bool = True
    java_initial_heap_mb: int = 512
    java_max_heap_mb: int = 8192
    java_disable_d3d: bool = True
    java_disable_directdraw: bool = True
    below_normal_process_priority: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"

    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def bytes_to_mib(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return ""
    return f"{num_bytes / (1024.0 * 1024.0):.2f}"


def process_memory_bytes() -> dict[str, Optional[int]]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
                wintypes.DWORD,
            ]
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if ok:
                return {
                    "process_rss_bytes": int(counters.WorkingSetSize),
                    "process_private_bytes": int(counters.PrivateUsage),
                }
        except Exception:
            return {"process_rss_bytes": None, "process_private_bytes": None}

    try:
        import resource

        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = int(rss_kib)
        else:
            rss_bytes = int(rss_kib * 1024)
        return {"process_rss_bytes": rss_bytes, "process_private_bytes": None}
    except Exception:
        return {"process_rss_bytes": None, "process_private_bytes": None}


def java_memory_bytes() -> dict[str, Optional[int]]:
    try:
        from jnius import autoclass

        runtime = autoclass("java.lang.Runtime").getRuntime()
        total = int(runtime.totalMemory())
        free = int(runtime.freeMemory())
        return {
            "java_heap_used_bytes": total - free,
            "java_heap_total_bytes": total,
            "java_heap_max_bytes": int(runtime.maxMemory()),
        }
    except Exception:
        return {
            "java_heap_used_bytes": None,
            "java_heap_total_bytes": None,
            "java_heap_max_bytes": None,
        }


def cuda_memory_bytes(device: torch.device) -> dict[str, Optional[int]]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_max_allocated_bytes": None,
        }

    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def memory_stats(device: torch.device) -> dict[str, Optional[int]]:
    stats = {}
    stats.update(process_memory_bytes())
    stats.update(java_memory_bytes())
    stats.update(cuda_memory_bytes(device))
    return stats


def memory_metrics_row(stats: dict[str, Optional[int]]) -> dict[str, str]:
    return {
        "process_rss_mb": bytes_to_mib(stats.get("process_rss_bytes")),
        "process_private_mb": bytes_to_mib(stats.get("process_private_bytes")),
        "java_heap_used_mb": bytes_to_mib(stats.get("java_heap_used_bytes")),
        "java_heap_total_mb": bytes_to_mib(stats.get("java_heap_total_bytes")),
        "java_heap_max_mb": bytes_to_mib(stats.get("java_heap_max_bytes")),
        "cuda_allocated_mb": bytes_to_mib(stats.get("cuda_allocated_bytes")),
        "cuda_reserved_mb": bytes_to_mib(stats.get("cuda_reserved_bytes")),
        "cuda_max_allocated_mb": bytes_to_mib(stats.get("cuda_max_allocated_bytes")),
    }


def format_memory_summary(stats: dict[str, Optional[int]]) -> str:
    return (
        f"rss={format_bytes(stats.get('process_rss_bytes'))} "
        f"private={format_bytes(stats.get('process_private_bytes'))} "
        f"java={format_bytes(stats.get('java_heap_used_bytes'))}/"
        f"{format_bytes(stats.get('java_heap_total_bytes'))} "
        f"cuda={format_bytes(stats.get('cuda_allocated_bytes'))}/"
        f"{format_bytes(stats.get('cuda_reserved_bytes'))}"
    )


def append_java_tool_option(option: str) -> None:
    current = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if option not in current.split():
        os.environ["JAVA_TOOL_OPTIONS"] = f"{current} {option}".strip()


def remove_java_tool_option(option_prefix: str) -> None:
    current = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if not current:
        return

    kept = [option for option in current.split() if not option.startswith(option_prefix)]
    if kept:
        os.environ["JAVA_TOOL_OPTIONS"] = " ".join(kept)
    else:
        os.environ.pop("JAVA_TOOL_OPTIONS", None)


def replace_java_tool_option(option_prefix: str, option: str) -> None:
    remove_java_tool_option(option_prefix)
    append_java_tool_option(option)


def set_below_normal_process_priority() -> None:
    if os.name != "nt":
        return

    try:
        import ctypes

        below_normal_priority_class = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, below_normal_priority_class)
    except Exception as exc:
        print(f"WARNING: nie udalo sie ustawic nizszego priorytetu procesu: {exc}")


def configure_runtime(cfg: Config) -> None:
    if cfg.cpu_threads > 0:
        thread_count = str(cfg.cpu_threads)
        os.environ.setdefault("OMP_NUM_THREADS", thread_count)
        os.environ.setdefault("MKL_NUM_THREADS", thread_count)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", thread_count)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", thread_count)
        torch.set_num_threads(cfg.cpu_threads)

    if cfg.cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(cfg.cpu_interop_threads)
        except RuntimeError:
            pass

    if cfg.java_headless:
        append_java_tool_option("-Djava.awt.headless=true")
        try:
            import jnius_config

            jnius_config.add_options("-Djava.awt.headless=true")
        except Exception:
            pass
    else:
        remove_java_tool_option("-Djava.awt.headless")
        remove_java_tool_option("-XX:ActiveProcessorCount")

    if cfg.java_disable_d3d:
        append_java_tool_option("-Dsun.java2d.d3d=false")
    else:
        remove_java_tool_option("-Dsun.java2d.d3d")

    if cfg.java_disable_directdraw:
        append_java_tool_option("-Dsun.java2d.noddraw=true")
    else:
        remove_java_tool_option("-Dsun.java2d.noddraw")

    if cfg.java_initial_heap_mb > 0:
        replace_java_tool_option("-Xms", f"-Xms{cfg.java_initial_heap_mb}m")
    if cfg.java_max_heap_mb > 0:
        replace_java_tool_option("-Xmx", f"-Xmx{cfg.java_max_heap_mb}m")
    try:
        import jnius_config

        if cfg.java_initial_heap_mb > 0:
            jnius_config.add_options(f"-Xms{cfg.java_initial_heap_mb}m")
        if cfg.java_max_heap_mb > 0:
            jnius_config.add_options(f"-Xmx{cfg.java_max_heap_mb}m")
        if cfg.java_disable_d3d:
            jnius_config.add_options("-Dsun.java2d.d3d=false")
        if cfg.java_disable_directdraw:
            jnius_config.add_options("-Dsun.java2d.noddraw=true")
    except Exception:
        pass

    if cfg.below_normal_process_priority:
        set_below_normal_process_priority()


def import_registered_bot_module(module_path: Path) -> None:
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku Drefsante: {module_path}")

    module_name = f"internet_bot_{module_path.stem}_{abs(hash(str(module_path)))}"
    if module_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nie mozna zaladowac modulu z {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    old_cwd = Path.cwd()
    parent_str = str(module_path.parent)
    added_path = False
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        added_path = True

    try:
        os.chdir(module_path.parent)
        allow_redundant_jnius_classpath_updates()
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        if added_path:
            try:
                sys.path.remove(parent_str)
            except ValueError:
                pass


def load_policy_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_policy(policy_cls, spatial_shape, non_spatial_size: int, action_space: int):
    policy_kwargs: dict[str, Any] = {
        "spatial_shape": spatial_shape,
        "non_spatial_size": non_spatial_size,
        "action_space": action_space,
    }
    if "hidden_nodes" in inspect.signature(policy_cls.__init__).parameters:
        policy_kwargs["hidden_nodes"] = getattr(policy_cls, "recommended_hidden_nodes", 128)
    return policy_cls(**policy_kwargs)


class ImitationBuffer:
    def __init__(
        self,
        recent_maxlen: int,
        archive_maxlen: int,
        archive_sample_fraction: float,
    ) -> None:
        self.capacity = max(1, int(recent_maxlen))
        self.size = 0
        self.spatial_samples: Optional[np.ndarray] = None
        self.non_spatial_samples: Optional[np.ndarray] = None
        self.action_mask_samples: Optional[np.ndarray] = None
        self.actions: Optional[np.ndarray] = None
        self.replacements = 0
        self.total_appended = 0

    def __len__(self) -> int:
        return self.size

    @property
    def recent_size(self) -> int:
        return self.size

    @property
    def archive_size(self) -> int:
        return 0

    def _ensure_storage(
        self,
        spatial: np.ndarray,
        non_spatial: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        if self.spatial_samples is not None:
            return

        self.spatial_samples = np.empty(
            (self.capacity, *spatial.shape),
            dtype=np.float32,
        )
        self.non_spatial_samples = np.empty(
            (self.capacity, *non_spatial.shape),
            dtype=np.float32,
        )
        self.action_mask_samples = np.empty(
            (self.capacity, *action_mask.shape),
            dtype=np.bool_,
        )
        self.actions = np.empty(self.capacity, dtype=np.int64)

    def append(
        self,
        spatial: np.ndarray,
        non_spatial: np.ndarray,
        action_mask: np.ndarray,
        action_idx: int,
    ) -> None:
        self._ensure_storage(spatial, non_spatial, action_mask)
        assert self.spatial_samples is not None
        assert self.non_spatial_samples is not None
        assert self.action_mask_samples is not None
        assert self.actions is not None

        action_idx = int(action_idx)

        self.total_appended += 1
        if self.size < self.capacity:
            write_idx = self.size
            self.size += 1
        else:
            write_idx = random.randrange(self.capacity)
            self.replacements += 1

        np.copyto(self.spatial_samples[write_idx], spatial, casting="unsafe")
        np.copyto(self.non_spatial_samples[write_idx], non_spatial, casting="unsafe")
        np.copyto(self.action_mask_samples[write_idx], action_mask, casting="unsafe")
        self.actions[write_idx] = action_idx

    def sample(self, batch_size: int, device: torch.device):
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty imitation buffer")

        assert self.spatial_samples is not None
        assert self.non_spatial_samples is not None
        assert self.action_mask_samples is not None
        assert self.actions is not None

        indices = np.random.randint(0, self.size, size=batch_size)
        spatial = torch.from_numpy(self.spatial_samples[indices]).float().to(device)
        non_spatial = torch.from_numpy(self.non_spatial_samples[indices]).float().to(device)
        action_mask = torch.from_numpy(self.action_mask_samples[indices]).bool().to(device)
        actions = torch.from_numpy(self.actions[indices]).long().to(device)
        return spatial, non_spatial, action_mask, actions


class RecordingExpertAgent(botbowl.Agent):
    def __init__(self, name: str, bot_name: str, buffer: ImitationBuffer):
        super().__init__(name)
        self.agent = botbowl.make_bot(bot_name)
        self.agent.name = name
        self.buffer = buffer
        self.env = None
        self.recorded_actions = 0
        self.skipped_actions = 0

    def new_game(self, game, team):
        return self.agent.new_game(game, team)

    def act(self, game):
        action = self.agent.act(game)
        if self.env is None or action is None:
            return action

        try:
            spatial, non_spatial, action_mask = self.env.get_state()
            action_idx = self.env._compute_action_idx(action)
            if bool(action_mask[action_idx]):
                self.buffer.append(spatial, non_spatial, action_mask, action_idx)
                self.recorded_actions += 1
            else:
                self.skipped_actions += 1
        except Exception:
            self.skipped_actions += 1
        return action

    def end_game(self, game):
        try:
            return self.agent.end_game(game)
        except PermissionError as exc:
            print(f"WARNING: Drefsante cleanup skipped locked temp file: {exc}")
            return None
        finally:
            cleanup_drefsante_agent(self.agent)


class SafeDrefsanteAgent(botbowl.Agent):
    def __init__(self, name: str, bot_name: str):
        super().__init__(name)
        self.agent = botbowl.make_bot(bot_name)
        self.agent.name = name

    def new_game(self, game, team):
        return self.agent.new_game(game, team)

    def act(self, game):
        return self.agent.act(game)

    def end_game(self, game):
        try:
            return self.agent.end_game(game)
        except PermissionError as exc:
            print(f"WARNING: Drefsante cleanup skipped locked temp file: {exc}")
            return None
        finally:
            cleanup_drefsante_agent(self.agent)


def make_drefsante_agent(name: str = DREFSANTE_BOT_NAME):
    return SafeDrefsanteAgent(name, name)


def make_env(env_size: int, away_agent):
    from botbowl.ai.env import BotBowlEnv, EnvConf, RewardWrapper

    env = BotBowlEnv(
        EnvConf(size=env_size, pathfinding=False),
        home_agent="human",
        away_agent=away_agent,
    )
    return RewardWrapper(env, home_reward_func=A2C_Reward(use_turn_end_rewards=False))


def root_env(env):
    return env.root_env if hasattr(env, "root_env") else env


def safe_end_current_game(env) -> None:
    root = root_env(env)
    game = getattr(root, "game", None)
    if game is None or getattr(game.state, "game_over", False):
        return

    for agent in (getattr(game, "home_agent", None), getattr(game, "away_agent", None)):
        if agent is not None and not getattr(agent, "human", True):
            try:
                agent.end_game(game)
            except Exception:
                pass


def safe_env_reset(env):
    safe_end_current_game(env)
    return env.reset()


def current_env_state(env):
    return root_env(env).get_state()


def compute_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    t_steps = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    running_return = bootstrap_value
    for t in reversed(range(t_steps)):
        running_return = rewards[t] + gamma * running_return * (1.0 - dones[t])
        returns[t] = running_return
    return returns


def masked_cross_entropy(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~action_mask, -1e9)
    return F.cross_entropy(masked_logits, actions)


def as_float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).float().to(device)


def as_bool_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).bool().to(device)


def init_metrics_file(path: Path) -> None:
    if path.exists():
        return

    fieldnames = [
        "phase",
        "iteration",
        "elapsed_sec",
        "phase_elapsed_sec",
        "samples",
        "recent_samples",
        "archive_samples",
        "total_recorded_samples",
        "expert_recorded",
        "expert_skipped",
        "timesteps",
        "episodes_finished_total",
        "wins_total",
        "losses_total",
        "draws_total",
        "win_rate_total",
        "mean_episode_return_20",
        "mean_td_for_20",
        "mean_td_opponent_20",
        "imitation_loss",
        "value_loss",
        "policy_loss",
        "policy_entropy",
        "process_rss_mb",
        "process_private_mb",
        "java_heap_used_mb",
        "java_heap_total_mb",
        "java_heap_max_mb",
        "cuda_allocated_mb",
        "cuda_reserved_mb",
        "cuda_max_allocated_mb",
    ]
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    policy,
    optimizer,
    cfg: Config,
    phase: str,
    iteration: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
        "phase": phase,
        "iteration": iteration,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_imitation_phase(
    cfg: Config,
    policy,
    optimizer,
    env,
    recorder: RecordingExpertAgent,
    buffer: ImitationBuffer,
    device: torch.device,
    metrics_path: Path,
    checkpoint_path: Path,
    global_started_at: float,
) -> None:
    phase_seconds = cfg.imitation_hours * 3600.0
    checkpoint_seconds = cfg.checkpoint_interval_minutes * 60.0
    phase_started_at = time.time()
    last_checkpoint_at = phase_started_at
    iteration = 0
    imitation_loss_value = 0.0

    spatial_np, non_spatial_np, action_mask_np = current_env_state(env)
    recorder.env = root_env(env)

    print(f"Start imitation learning: {cfg.imitation_hours:.2f}h")
    while time.time() - phase_started_at < phase_seconds:
        iteration += 1
        spatial = as_float_tensor(spatial_np[None], device)
        non_spatial = as_float_tensor(non_spatial_np[None], device)
        action_mask = as_bool_tensor(action_mask_np[None], device)

        with torch.no_grad():
            _, actions = policy.act(spatial, non_spatial, action_mask)
        action_idx = int(actions.item())

        (next_spatial, next_non_spatial, next_mask), _, done, _ = env.step(action_idx)
        if done:
            spatial_np, non_spatial_np, action_mask_np = safe_env_reset(env)
            recorder.env = root_env(env)
        else:
            spatial_np, non_spatial_np, action_mask_np = next_spatial, next_non_spatial, next_mask

        if len(buffer) >= cfg.imitation_batch_size:
            losses = []
            for _ in range(cfg.imitation_train_batches_per_cycle):
                batch_spatial, batch_non_spatial, batch_mask, batch_actions = buffer.sample(
                    cfg.imitation_batch_size,
                    device,
                )
                _, logits = policy(batch_spatial, batch_non_spatial)
                loss = masked_cross_entropy(logits, batch_mask, batch_actions)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                losses.append(loss.detach())
            imitation_loss_value = float(torch.stack(losses).mean().item())

        if iteration % cfg.log_interval == 0:
            if len(buffer) >= buffer.capacity:
                gc.collect()
            mem_stats = memory_stats(device)
            phase_elapsed = time.time() - phase_started_at
            elapsed = time.time() - global_started_at
            print(
                f"IL iter={iteration} elapsed={format_duration(phase_elapsed)} "
                f"samples={len(buffer)} recent={buffer.recent_size} "
                f"archive={buffer.archive_size} loss={imitation_loss_value:.4f} "
                f"{format_memory_summary(mem_stats)}"
            )
            row = {
                "phase": "imitation",
                "iteration": iteration,
                "elapsed_sec": elapsed,
                "phase_elapsed_sec": phase_elapsed,
                "samples": len(buffer),
                "recent_samples": buffer.recent_size,
                "archive_samples": buffer.archive_size,
                "total_recorded_samples": buffer.total_appended,
                "expert_recorded": recorder.recorded_actions,
                "expert_skipped": recorder.skipped_actions,
                "timesteps": iteration,
                "episodes_finished_total": "",
                "wins_total": "",
                "losses_total": "",
                "draws_total": "",
                "win_rate_total": "",
                "mean_episode_return_20": "",
                "mean_td_for_20": "",
                "mean_td_opponent_20": "",
                "imitation_loss": imitation_loss_value,
                "value_loss": "",
                "policy_loss": "",
                "policy_entropy": "",
            }
            row.update(memory_metrics_row(mem_stats))
            append_metrics(metrics_path, row)

        if time.time() - last_checkpoint_at >= checkpoint_seconds:
            save_checkpoint(
                checkpoint_path,
                policy,
                optimizer,
                cfg,
                phase="imitation",
                iteration=iteration,
                extra={
                    "imitation_samples": len(buffer),
                    "recent_samples": buffer.recent_size,
                    "archive_samples": buffer.archive_size,
                    "total_recorded_samples": buffer.total_appended,
                },
            )
            last_checkpoint_at = time.time()
            print(f"Saved imitation checkpoint: {checkpoint_path}")

    save_checkpoint(
        checkpoint_path,
        policy,
        optimizer,
        cfg,
        phase="imitation",
        iteration=iteration,
        extra={
            "imitation_samples": len(buffer),
            "recent_samples": buffer.recent_size,
            "archive_samples": buffer.archive_size,
            "total_recorded_samples": buffer.total_appended,
        },
    )
    print(f"Finished imitation phase. Saved checkpoint: {checkpoint_path}")


def run_reinforcement_phase(
    cfg: Config,
    policy,
    optimizer,
    env,
    device: torch.device,
    metrics_path: Path,
    best_ckpt: Path,
    final_ckpt: Path,
    global_started_at: float,
) -> None:
    phase_seconds = cfg.reinforcement_hours * 3600.0
    checkpoint_seconds = cfg.checkpoint_interval_minutes * 60.0
    phase_started_at = time.time()
    last_checkpoint_at = phase_started_at
    iteration = 0
    timesteps = 0

    spatial_np, non_spatial_np, action_mask_np = current_env_state(env)
    spatial_shape = spatial_np.shape
    non_spatial_size = int(non_spatial_np.shape[0])
    action_space = int(action_mask_np.shape[0])

    episode_return = 0.0
    episode_td_for = 0.0
    episode_td_opponent = 0.0
    recent_returns = deque(maxlen=20)
    recent_td_for = deque(maxlen=20)
    recent_td_opponent = deque(maxlen=20)
    recent_outcomes = deque(maxlen=20)
    wins_total = losses_total = draws_total = episodes_finished_total = 0
    best_score = None

    print(f"Start reinforcement learning vs Drefsante: {cfg.reinforcement_hours:.2f}h")
    while time.time() - phase_started_at < phase_seconds:
        iteration += 1
        rollout_spatial = torch.zeros(cfg.rollout_len + 1, 1, *spatial_shape, device=device)
        rollout_non_spatial = torch.zeros(cfg.rollout_len + 1, 1, non_spatial_size, device=device)
        rollout_masks = torch.zeros(cfg.rollout_len + 1, 1, action_space, dtype=torch.bool, device=device)
        rollout_actions = torch.zeros(cfg.rollout_len, 1, 1, dtype=torch.long, device=device)
        rollout_rewards = torch.zeros(cfg.rollout_len, 1, 1, device=device)
        rollout_dones = torch.zeros(cfg.rollout_len, 1, 1, device=device)
        rollout_values = torch.zeros(cfg.rollout_len, 1, 1, device=device)

        rollout_spatial[0, 0].copy_(as_float_tensor(spatial_np, device))
        rollout_non_spatial[0, 0].copy_(as_float_tensor(non_spatial_np, device))
        rollout_masks[0, 0].copy_(as_bool_tensor(action_mask_np, device))

        for step in range(cfg.rollout_len):
            with torch.no_grad():
                values, actions = policy.act(
                    rollout_spatial[step],
                    rollout_non_spatial[step],
                    rollout_masks[step],
                )

            root = root_env(env)
            td_for_before = root.game.state.home_team.state.score
            td_opponent_before = root.game.state.away_team.state.score
            (next_spatial, next_non_spatial, next_mask), reward, done, _ = env.step(int(actions.item()))
            timesteps += 1

            td_for_after = root.game.state.home_team.state.score
            td_opponent_after = root.game.state.away_team.state.score
            episode_return += float(reward)
            episode_td_for += float(td_for_after - td_for_before)
            episode_td_opponent += float(td_opponent_after - td_opponent_before)

            if done:
                episodes_finished_total += 1
                recent_returns.append(episode_return)
                recent_td_for.append(episode_td_for)
                recent_td_opponent.append(episode_td_opponent)
                if episode_td_for > episode_td_opponent:
                    wins_total += 1
                    recent_outcomes.append("W")
                elif episode_td_for < episode_td_opponent:
                    losses_total += 1
                    recent_outcomes.append("L")
                else:
                    draws_total += 1
                    recent_outcomes.append("D")
                episode_return = episode_td_for = episode_td_opponent = 0.0
                next_spatial, next_non_spatial, next_mask = safe_env_reset(env)

            rollout_actions[step].copy_(actions)
            rollout_values[step].copy_(values)
            rollout_rewards[step, 0, 0] = float(reward)
            rollout_dones[step, 0, 0] = float(done)
            rollout_spatial[step + 1, 0].copy_(as_float_tensor(next_spatial, device))
            rollout_non_spatial[step + 1, 0].copy_(as_float_tensor(next_non_spatial, device))
            rollout_masks[step + 1, 0].copy_(as_bool_tensor(next_mask, device))
            spatial_np, non_spatial_np, action_mask_np = next_spatial, next_non_spatial, next_mask

        with torch.no_grad():
            next_values, _ = policy(rollout_spatial[-1], rollout_non_spatial[-1])

        returns = compute_discounted_returns(
            rollout_rewards,
            rollout_dones,
            next_values,
            cfg.gamma,
        )
        advantages = returns - rollout_values
        batch_size = cfg.rollout_len
        flat_spatial = rollout_spatial[:-1].reshape(batch_size, *spatial_shape)
        flat_non_spatial = rollout_non_spatial[:-1].reshape(batch_size, non_spatial_size)
        flat_masks = rollout_masks[:-1].reshape(batch_size, action_space)
        flat_actions = rollout_actions.reshape(batch_size, 1)
        flat_returns = returns.reshape(batch_size, 1)
        flat_advantages = advantages.reshape(batch_size, 1)

        new_log_probs, new_values, policy_entropy = policy.evaluate_actions(
            flat_spatial,
            flat_non_spatial,
            flat_actions,
            flat_masks,
        )
        policy_loss = -(flat_advantages.detach() * new_log_probs).mean()
        value_loss = F.mse_loss(new_values, flat_returns)
        loss = cfg.value_loss_coef * value_loss + policy_loss - cfg.entropy_coef * policy_entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()

        all_games = wins_total + losses_total + draws_total
        win_rate_total = (wins_total / all_games) if all_games > 0 else 0.0
        mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
        mean_td_for = float(np.mean(recent_td_for)) if recent_td_for else 0.0
        mean_td_opponent = float(np.mean(recent_td_opponent)) if recent_td_opponent else 0.0

        current_score = (win_rate_total, mean_return, mean_td_for - mean_td_opponent)
        if best_score is None or current_score > best_score:
            best_score = current_score
            save_checkpoint(
                best_ckpt,
                policy,
                optimizer,
                cfg,
                phase="reinforcement",
                iteration=iteration,
                extra={"timesteps": timesteps, "best_score": best_score},
            )
            print(f"Saved best RL checkpoint: {best_ckpt}")

        if iteration % cfg.log_interval == 0:
            mem_stats = memory_stats(device)
            phase_elapsed = time.time() - phase_started_at
            elapsed = time.time() - global_started_at
            print(
                f"RL update={iteration} elapsed={format_duration(phase_elapsed)} "
                f"games={all_games} win_rate={win_rate_total:.3f} "
                f"policy_loss={policy_loss.item():.4f} "
                f"{format_memory_summary(mem_stats)}"
            )
            row = {
                "phase": "reinforcement",
                "iteration": iteration,
                "elapsed_sec": elapsed,
                "phase_elapsed_sec": phase_elapsed,
                "samples": "",
                "recent_samples": "",
                "archive_samples": "",
                "total_recorded_samples": "",
                "expert_recorded": "",
                "expert_skipped": "",
                "timesteps": timesteps,
                "episodes_finished_total": episodes_finished_total,
                "wins_total": wins_total,
                "losses_total": losses_total,
                "draws_total": draws_total,
                "win_rate_total": win_rate_total,
                "mean_episode_return_20": mean_return,
                "mean_td_for_20": mean_td_for,
                "mean_td_opponent_20": mean_td_opponent,
                "imitation_loss": "",
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "policy_entropy": policy_entropy.item(),
            }
            row.update(memory_metrics_row(mem_stats))
            append_metrics(metrics_path, row)

        if time.time() - last_checkpoint_at >= checkpoint_seconds:
            save_checkpoint(
                final_ckpt,
                policy,
                optimizer,
                cfg,
                phase="reinforcement",
                iteration=iteration,
                extra={"timesteps": timesteps},
            )
            last_checkpoint_at = time.time()
            print(f"Saved periodic RL checkpoint: {final_ckpt}")

    save_checkpoint(
        final_ckpt,
        policy,
        optimizer,
        cfg,
        phase="reinforcement",
        iteration=iteration,
        extra={"timesteps": timesteps},
    )
    print(f"Finished RL phase. Saved final checkpoint: {final_ckpt}")


def main() -> None:
    cfg = Config()
    configure_runtime(cfg)
    verify_drefsante_java_class()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_size = resolve_env_size(cfg.env_size_default)

    import_registered_bot_module(Path(__file__).resolve().parent / DREFSANTE_MODULE_PATH)

    buffer = ImitationBuffer(
        recent_maxlen=cfg.imitation_recent_buffer_size,
        archive_maxlen=cfg.imitation_archive_buffer_size,
        archive_sample_fraction=cfg.imitation_archive_sample_fraction,
    )
    recorder = RecordingExpertAgent("Drefsante recorder", DREFSANTE_BOT_NAME, buffer)
    il_env = make_env(env_size, away_agent=recorder)
    recorder.env = root_env(il_env)

    spatial_np, non_spatial_np, action_mask_np = current_env_state(il_env)
    recorder.env = root_env(il_env)

    policy_cls = load_policy_class(cfg.policy_module, cfg.policy_class)
    policy = build_policy(
        policy_cls,
        spatial_shape=tuple(spatial_np.shape),
        non_spatial_size=int(non_spatial_np.shape[0]),
        action_space=int(action_mask_np.shape[0]),
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.imitation_lr)

    run_name, out_dir, metrics_path, best_ckpt, final_ckpt, run_tag = build_unique_run_paths(
        out_dir_root=cfg.out_dir_root,
        policy_module=cfg.policy_module,
        script_path=__file__,
        env_size=env_size,
        metrics_prefix="drefsante_curriculum_metrics",
        checkpoint_prefix="drefsante_curriculum",
    )
    imitation_ckpt = out_dir / f"drefsante_curriculum__{run_tag}_after_imitation.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    init_metrics_file(metrics_path)

    print(f"Device={device}, env_size={env_size}")
    print(
        f"CPU threads={cfg.cpu_threads}, interop_threads={cfg.cpu_interop_threads}, "
        f"java_headless={cfg.java_headless}"
    )
    print(
        f"Java heap: Xms={cfg.java_initial_heap_mb}m, "
        f"Xmx={cfg.java_max_heap_mb}m"
    )
    print(f"Run name: {run_name}")
    print(f"Metrics CSV: {metrics_path}")
    print(f"Imitation checkpoint: {imitation_ckpt}")
    print(f"Best RL checkpoint: {best_ckpt}")
    print(f"Final checkpoint: {final_ckpt}")

    global_started_at = time.time()
    try:
        run_imitation_phase(
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            env=il_env,
            recorder=recorder,
            buffer=buffer,
            device=device,
            metrics_path=metrics_path,
            checkpoint_path=imitation_ckpt,
            global_started_at=global_started_at,
        )
    finally:
        safe_end_current_game(il_env)
        il_env.close()

    optimizer = optim.Adam(policy.parameters(), lr=cfg.reinforcement_lr)
    rl_env = make_env(env_size, away_agent=make_drefsante_agent())
    try:
        run_reinforcement_phase(
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            env=rl_env,
            device=device,
            metrics_path=metrics_path,
            best_ckpt=best_ckpt,
            final_ckpt=final_ckpt,
            global_started_at=global_started_at,
        )
    finally:
        safe_end_current_game(rl_env)
        rl_env.close()


if __name__ == "__main__":
    main()
