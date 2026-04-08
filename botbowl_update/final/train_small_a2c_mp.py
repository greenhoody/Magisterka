from __future__ import annotations

import csv
import importlib
import inspect
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from botbowl.ai.env import BotBowlEnv, EnvConf, RewardWrapper
from a2c_env import A2C_Reward
from training_env import resolve_env_size


@dataclass
class Config:
    seed: int = 42
    env_size_default: int = 5
    num_envs: int = 8  # >=2 przez BatchNorm w CustomPolicy
    num_steps: int = 2_000_000
    rollout_len: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    log_interval: int = 10
    save_interval: int = 50
    policy_module: str = "small_network_inception_block"
    policy_class: str = "CustomPolicy"
    out_dir_root: str = "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_env(env_size: int) -> BotBowlEnv:
    env = BotBowlEnv(EnvConf(size=env_size, pathfinding=False))
    return RewardWrapper(env, home_reward_func=A2C_Reward(use_turn_end_rewards=True))


def load_policy_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def compute_gae(
    rewards: torch.Tensor,  # [T, N, 1]
    dones: torch.Tensor,  # [T, N, 1], 1.0 = done
    values: torch.Tensor,  # [T+1, N, 1]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_steps, n_envs, _ = rewards.shape
    advantages = torch.zeros(t_steps, n_envs, 1, device=rewards.device)
    gae = torch.zeros(n_envs, 1, device=rewards.device)

    for t in reversed(range(t_steps)):
        cont_mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * cont_mask - values[t]
        gae = delta + gamma * lam * cont_mask * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return returns, advantages


def compute_explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    y_pred = values.detach().reshape(-1)
    y_true = returns.detach().reshape(-1)
    y_var = torch.var(y_true, unbiased=False)
    if y_var <= 1e-8:
        return 0.0
    return (1.0 - torch.var(y_true - y_pred, unbiased=False) / y_var).item()


def compute_action_diagnostics(
    actions: torch.Tensor, action_masks: torch.Tensor, action_space: int
) -> dict:
    flat_actions = actions.detach().reshape(-1).long()
    counts = torch.bincount(flat_actions, minlength=action_space).float()
    total = max(int(counts.sum().item()), 1)

    used_actions = int((counts > 0).sum().item())
    top1_frac = (counts.max().item() / total) if total > 0 else 0.0

    top_k = min(5, action_space)
    topk_counts, topk_indices = torch.topk(counts, k=top_k)
    top5_frac = (topk_counts.sum().item() / total) if total > 0 else 0.0

    probs = counts / float(total)
    nz = probs > 0
    entropy = -(probs[nz] * torch.log(probs[nz])).sum().item()
    entropy_norm = entropy / float(np.log(action_space)) if action_space > 1 else 0.0

    chosen_is_valid = (
        action_masks[:-1].gather(dim=2, index=actions.long()).float().mean().item()
    )
    mean_valid_actions = action_masks[:-1].sum(dim=2).float().mean().item()

    top_actions = []
    for idx, cnt in zip(topk_indices.tolist(), topk_counts.tolist()):
        frac = cnt / total if total > 0 else 0.0
        top_actions.append(f"{idx}:{frac:.3f}")

    return {
        "action_unique_frac": used_actions / float(action_space),
        "action_top1_frac": float(top1_frac),
        "action_top5_frac": float(top5_frac),
        "action_entropy_norm": float(entropy_norm),
        "chosen_action_valid_rate": float(chosen_is_valid),
        "mean_valid_actions": float(mean_valid_actions),
        "mean_valid_action_ratio": float(mean_valid_actions / float(action_space)),
        "top_actions": "|".join(top_actions),
    }


def worker(remote, parent_remote, env_size: int):
    parent_remote.close()
    env = make_env(env_size)
    td_for = 0
    td_opponent = 0

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action = int(data)
                (spatial, non_spatial, action_mask), reward, done, _ = env.step(action)

                game = env.game
                current_td_for = game.state.home_team.state.score
                current_td_opponent = game.state.away_team.state.score
                td_for_scored = current_td_for - td_for
                td_opponent_scored = current_td_opponent - td_opponent
                td_for = current_td_for
                td_opponent = current_td_opponent

                if done:
                    next_spatial, next_non_spatial, next_action_mask = env.reset()
                    td_for = 0
                    td_opponent = 0
                else:
                    next_spatial, next_non_spatial, next_action_mask = (
                        spatial,
                        non_spatial,
                        action_mask,
                    )

                remote.send(
                    (
                        next_spatial,
                        next_non_spatial,
                        next_action_mask,
                        float(reward),
                        bool(done),
                        int(td_for_scored),
                        int(td_opponent_scored),
                    )
                )
            elif cmd == "reset":
                spatial, non_spatial, action_mask = env.reset()
                td_for = 0
                td_opponent = 0
                remote.send((spatial, non_spatial, action_mask))
            elif cmd == "close":
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs: int, env_size: int):
        self.closed = False
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])

        self.ps = [
            Process(target=worker, args=(work_remote, remote, env_size))
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        for work_remote in self.work_remotes:
            work_remote.close()

    def step(self, actions: Iterable[int]):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        spatial, non_spatial, action_mask, rewards, dones, td_for, td_opponent = zip(
            *results
        )
        return (
            np.stack(spatial, axis=0),
            np.stack(non_spatial, axis=0),
            np.stack(action_mask, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            np.asarray(td_for, dtype=np.int32),
            np.asarray(td_opponent, dtype=np.int32),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        spatial, non_spatial, action_mask = zip(*results)
        return (
            np.stack(spatial, axis=0),
            np.stack(non_spatial, axis=0),
            np.stack(action_mask, axis=0),
        )

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


class RolloutMemory:
    def __init__(
        self,
        rollout_len: int,
        num_envs: int,
        spatial_shape,
        non_spatial_size: int,
        action_space: int,
        device: torch.device,
    ):
        t_steps, n_envs = rollout_len, num_envs
        self.spatial_obs = torch.zeros(
            t_steps + 1, n_envs, *spatial_shape, device=device
        )
        self.non_spatial_obs = torch.zeros(
            t_steps + 1, n_envs, non_spatial_size, device=device
        )
        self.action_masks = torch.zeros(
            t_steps + 1, n_envs, action_space, dtype=torch.bool, device=device
        )

        self.actions = torch.zeros(t_steps, n_envs, 1, dtype=torch.long, device=device)
        self.rewards = torch.zeros(t_steps, n_envs, 1, device=device)
        self.dones = torch.zeros(t_steps, n_envs, 1, device=device)
        self.values = torch.zeros(t_steps, n_envs, 1, device=device)


def init_metrics_file(path: Path) -> None:
    if path.exists():
        return

    fieldnames = [
        "update",
        "total_updates",
        "timesteps",
        "progress_pct",
        "elapsed_sec",
        "eta_sec",
        "timesteps_per_sec",
        "episodes_finished_total",
        "wins_total",
        "losses_total",
        "draws_total",
        "win_rate_total",
        "mean_episode_return_50",
        "mean_td_for_50",
        "mean_td_opponent_50",
        "wins_50",
        "losses_50",
        "draws_50",
        "win_rate_50",
        "value_loss",
        "policy_loss",
        "policy_entropy",
        "explained_variance",
        "action_unique_frac",
        "action_top1_frac",
        "action_top5_frac",
        "action_entropy_norm",
        "chosen_action_valid_rate",
        "mean_valid_actions",
        "mean_valid_action_ratio",
        "top_actions",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_size = resolve_env_size(cfg.env_size_default)

    envs = VecEnv(num_envs=cfg.num_envs, env_size=env_size)
    spatial_np, non_spatial_np, mask_np = envs.reset()

    spatial_shape = spatial_np.shape[1:]
    non_spatial_size = non_spatial_np.shape[1]
    action_space = mask_np.shape[1]

    policy_cls = load_policy_class(cfg.policy_module, cfg.policy_class)
    policy_kwargs = {
        "spatial_shape": spatial_shape,
        "non_spatial_size": non_spatial_size,
        "action_space": action_space,
    }
    if "hidden_nodes" in inspect.signature(policy_cls.__init__).parameters:
        policy_kwargs["hidden_nodes"] = action_space
    policy = policy_cls(**policy_kwargs).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    memory = RolloutMemory(
        rollout_len=cfg.rollout_len,
        num_envs=cfg.num_envs,
        spatial_shape=spatial_shape,
        non_spatial_size=non_spatial_size,
        action_space=action_space,
        device=device,
    )

    memory.spatial_obs[0].copy_(torch.from_numpy(spatial_np).float().to(device))
    memory.non_spatial_obs[0].copy_(torch.from_numpy(non_spatial_np).float().to(device))
    memory.action_masks[0].copy_(torch.from_numpy(mask_np).to(device).bool())

    run_name = f"{cfg.policy_module}__{Path(__file__).stem}"
    out_dir = Path(cfg.out_dir_root) / run_name / f"botbowl-{env_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "training_metrics.csv"
    init_metrics_file(metrics_path)

    total_updates = cfg.num_steps // (cfg.rollout_len * cfg.num_envs)
    training_started_at = time.time()

    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    episode_td_for = np.zeros(cfg.num_envs, dtype=np.float32)
    episode_td_opponent = np.zeros(cfg.num_envs, dtype=np.float32)

    recent_returns = deque(maxlen=50)
    recent_td_for = deque(maxlen=50)
    recent_td_opponent = deque(maxlen=50)
    recent_outcomes = deque(maxlen=50)  # "W", "L", "D"

    wins_total = 0
    losses_total = 0
    draws_total = 0
    episodes_finished_total = 0

    print(
        f"Device={device}, env_size={env_size}, action_space={action_space}, "
        f"num_envs={cfg.num_envs}, rollout_len={cfg.rollout_len}"
    )
    print(f"Metrics CSV: {metrics_path}")

    try:
        for update in range(1, total_updates + 1):
            for step in range(cfg.rollout_len):
                with torch.no_grad():
                    values, actions = policy.act(
                        memory.spatial_obs[step],
                        memory.non_spatial_obs[step],
                        memory.action_masks[step],
                    )

                actions_np = actions.squeeze(1).cpu().numpy()
                (
                    next_spatial_np,
                    next_non_spatial_np,
                    next_mask_np,
                    rewards_np,
                    dones_np,
                    td_for_np,
                    td_opponent_np,
                ) = envs.step(actions_np)

                episode_returns += rewards_np
                episode_td_for += td_for_np
                episode_td_opponent += td_opponent_np

                for i in range(cfg.num_envs):
                    if dones_np[i]:
                        episodes_finished_total += 1
                        recent_returns.append(float(episode_returns[i]))
                        recent_td_for.append(float(episode_td_for[i]))
                        recent_td_opponent.append(float(episode_td_opponent[i]))

                        if episode_td_for[i] > episode_td_opponent[i]:
                            wins_total += 1
                            recent_outcomes.append("W")
                        elif episode_td_for[i] < episode_td_opponent[i]:
                            losses_total += 1
                            recent_outcomes.append("L")
                        else:
                            draws_total += 1
                            recent_outcomes.append("D")

                        episode_returns[i] = 0.0
                        episode_td_for[i] = 0.0
                        episode_td_opponent[i] = 0.0

                memory.actions[step].copy_(actions)
                memory.values[step].copy_(values)
                memory.rewards[step, :, 0].copy_(
                    torch.from_numpy(rewards_np).to(device)
                )
                memory.dones[step, :, 0].copy_(
                    torch.from_numpy(dones_np.astype(np.float32)).to(device)
                )

                memory.spatial_obs[step + 1].copy_(
                    torch.from_numpy(next_spatial_np).float().to(device)
                )
                memory.non_spatial_obs[step + 1].copy_(
                    torch.from_numpy(next_non_spatial_np).float().to(device)
                )
                memory.action_masks[step + 1].copy_(
                    torch.from_numpy(next_mask_np).to(device).bool()
                )

            with torch.no_grad():
                next_values, _ = policy(
                    memory.spatial_obs[-1], memory.non_spatial_obs[-1]
                )

            values_boot = torch.zeros(
                cfg.rollout_len + 1, cfg.num_envs, 1, device=device
            )
            values_boot[:-1].copy_(memory.values)
            values_boot[-1].copy_(next_values)

            returns, advantages = compute_gae(
                memory.rewards,
                memory.dones,
                values_boot,
                cfg.gamma,
                cfg.gae_lambda,
            )
            explained_variance = compute_explained_variance(memory.values, returns)
            action_diag = compute_action_diagnostics(
                memory.actions, memory.action_masks, action_space
            )
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            batch_size = cfg.rollout_len * cfg.num_envs
            flat_spatial = memory.spatial_obs[:-1].reshape(batch_size, *spatial_shape)
            flat_non_spatial = memory.non_spatial_obs[:-1].reshape(
                batch_size, non_spatial_size
            )
            flat_action_masks = memory.action_masks[:-1].reshape(
                batch_size, action_space
            )
            flat_actions = memory.actions.reshape(batch_size, 1)
            flat_returns = returns.reshape(batch_size, 1)
            flat_advantages = advantages.reshape(batch_size, 1)

            new_log_probs, new_values, policy_entropy = policy.evaluate_actions(
                flat_spatial,
                flat_non_spatial,
                flat_actions,
                flat_action_masks,
            )
            policy_loss = -(flat_advantages.detach() * new_log_probs).mean()
            value_loss = F.mse_loss(new_values, flat_returns)
            loss = (
                cfg.value_loss_coef * value_loss
                + policy_loss
                - cfg.entropy_coef * policy_entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            memory.spatial_obs[0].copy_(memory.spatial_obs[-1])
            memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
            memory.action_masks[0].copy_(memory.action_masks[-1])

            value_loss_mean = value_loss.detach().item()
            policy_loss_mean = policy_loss.detach().item()
            policy_entropy_mean = policy_entropy.detach().item()

            all_games = wins_total + losses_total + draws_total
            win_rate_total = (wins_total / all_games) if all_games > 0 else 0.0

            wins_50 = sum(1 for x in recent_outcomes if x == "W")
            losses_50 = sum(1 for x in recent_outcomes if x == "L")
            draws_50 = sum(1 for x in recent_outcomes if x == "D")
            games_50 = wins_50 + losses_50 + draws_50
            win_rate_50 = (wins_50 / games_50) if games_50 > 0 else 0.0

            mean_episode_return_50 = (
                float(np.mean(recent_returns)) if recent_returns else 0.0
            )
            mean_td_for_50 = float(np.mean(recent_td_for)) if recent_td_for else 0.0
            mean_td_opponent_50 = (
                float(np.mean(recent_td_opponent)) if recent_td_opponent else 0.0
            )

            if update % cfg.log_interval == 0:
                timesteps = update * cfg.rollout_len * cfg.num_envs
                progress_pct = 100.0 * update / total_updates if total_updates > 0 else 100.0
                elapsed_sec = time.time() - training_started_at
                avg_update_time = elapsed_sec / update if update > 0 else 0.0
                eta_sec = avg_update_time * max(total_updates - update, 0)
                timesteps_per_sec = timesteps / max(elapsed_sec, 1e-8)
                print(
                    f"update={update}/{total_updates} "
                    f"progress={progress_pct:.1f}% "
                    f"elapsed={format_duration(elapsed_sec)} "
                    f"eta={format_duration(eta_sec)}"
                )

                row = {
                    "update": update,
                    "total_updates": total_updates,
                    "timesteps": timesteps,
                    "progress_pct": progress_pct,
                    "elapsed_sec": elapsed_sec,
                    "eta_sec": eta_sec,
                    "timesteps_per_sec": timesteps_per_sec,
                    "episodes_finished_total": episodes_finished_total,
                    "wins_total": wins_total,
                    "losses_total": losses_total,
                    "draws_total": draws_total,
                    "win_rate_total": win_rate_total,
                    "mean_episode_return_50": mean_episode_return_50,
                    "mean_td_for_50": mean_td_for_50,
                    "mean_td_opponent_50": mean_td_opponent_50,
                    "wins_50": wins_50,
                    "losses_50": losses_50,
                    "draws_50": draws_50,
                    "win_rate_50": win_rate_50,
                    "value_loss": value_loss_mean,
                    "policy_loss": policy_loss_mean,
                    "policy_entropy": policy_entropy_mean,
                    "explained_variance": explained_variance,
                    "action_unique_frac": action_diag["action_unique_frac"],
                    "action_top1_frac": action_diag["action_top1_frac"],
                    "action_top5_frac": action_diag["action_top5_frac"],
                    "action_entropy_norm": action_diag["action_entropy_norm"],
                    "chosen_action_valid_rate": action_diag["chosen_action_valid_rate"],
                    "mean_valid_actions": action_diag["mean_valid_actions"],
                    "mean_valid_action_ratio": action_diag["mean_valid_action_ratio"],
                    "top_actions": action_diag["top_actions"],
                }
                append_metrics(metrics_path, row)

            if update % cfg.save_interval == 0:
                ckpt = out_dir / f"small_network_a2c_mp_upd{update}.pt"
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "config": asdict(cfg),
                        "update": update,
                    },
                    ckpt,
                )

        final_ckpt = out_dir / "small_network_a2c_mp_final.pt"
        torch.save(
            {
                "model": policy.state_dict(),
                "config": asdict(cfg),
            },
            final_ckpt,
        )
        print(f"Saved final checkpoint: {final_ckpt}")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
