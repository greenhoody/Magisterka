"""
Train the A2C Dynamic Hyper-Connection Spatial Inception policy with the random→checkpoint schedule.
"""

from functools import partial
from multiprocessing import Process, Pipe
import os
from typing import Iterable, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
from torch.serialization import add_safe_globals

import botbowl
from botbowl.ai.env import (
    BotBowlEnv,
    RewardWrapper,
    EnvConf,
    ScriptedActionWrapper,
    BotBowlWrapper,
    PPCGWrapper,
)
from a2c_dynamic_hyper_spatial_inception_agent import (
    A2CAgent,
    CNNPolicy,
    HyperConnection,
    HyperConnectionStack,
    InceptionBlock,
    SpatialInceptionBlock,
)
from a2c_env import A2C_Reward, a2c_scripted_actions
from botbowl.ai.layers import *
from checkpoint_scheduler import CheckpointOpponentScheduler
from training_env import resolve_env_size
from layer_norm import LayerNorm2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Allow torch.load to reconstruct policies saved from this process.
add_safe_globals(
    [
        CNNPolicy,
        InceptionBlock,
        SpatialInceptionBlock,
        nn.Sequential,
        nn.Conv2d,
        LayerNorm2d,
        nn.ReLU,
        nn.ModuleList,
        nn.Linear,
        nn.AdaptiveAvgPool2d,
        nn.PReLU,
        nn.LayerNorm,
        HyperConnection,
        HyperConnectionStack,
    ]
)


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


MODEL_KIND = "a2c-dynamic-hyper-random-checkpoint"

# Environment
env_size = resolve_env_size(5)  # Options are 1,3,5,7,11
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)

make_agent_from_model = partial(
    A2CAgent, env_conf=env_conf, scripted_func=a2c_scripted_actions
)


def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


# Training configuration
num_steps = 10000000
num_processes = 8
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
# entropy_coef = 0.01
entropy_coef = 0.04
value_loss_coef = 0.5
# max_grad_norm = 0.05
max_grad_norm = 0.5
log_interval = 50
# Model checkpoints are saved independently from log cadence to avoid disk spam.
save_interval = 1000
ppcg = False


reset_steps = 5000  # The environment is reset after this many steps it gets stuck

# Opponent schedule
min_updates_before_checkpoint = 1000
difficulty_threshold = 1.0
win_rate_threshold = 0.70

# Architecture
num_hidden_nodes = 128
num_residual_blocks = 6
num_cnn_kernels = [(18, 3), (18, 5), (18, 7)]

# When using A2CAgent, remember to set exclude_pathfinding_moves = False if you train with pathfinding_enabled = True


# Make directories
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir("logs/")
ensure_dir("models/")
ensure_dir("plots/")
run_id = f"{MODEL_KIND}_{timestamp_now()}"
log_dir = f"logs/{env_name}/"
model_dir = f"models/{env_name}/"
plot_dir = f"plots/{env_name}/"
ensure_dir(log_dir)
ensure_dir(model_dir)
ensure_dir(plot_dir)


class Memory(object):
    def __init__(
        self,
        steps_per_update,
        num_processes,
        spatial_obs_shape,
        non_spatial_obs_shape,
        action_space,
    ):
        self.spatial_obs = torch.zeros(
            steps_per_update + 1, num_processes, *spatial_obs_shape
        )
        self.non_spatial_obs = torch.zeros(
            steps_per_update + 1, num_processes, *non_spatial_obs_shape
        )
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(
            steps_per_update + 1, num_processes, action_space, dtype=torch.bool
        )

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(
        self, step, spatial_obs, non_spatial_obs, action, reward, mask, action_masks
    ):
        device = self.spatial_obs.device
        self.spatial_obs[step + 1].copy_(
            torch.from_numpy(spatial_obs).float().to(device)
        )
        self.non_spatial_obs[step + 1].copy_(
            torch.from_numpy(np.expand_dims(non_spatial_obs, axis=1)).float().to(device)
        )
        self.actions[step].copy_(action)
        self.rewards[step].copy_(
            torch.from_numpy(np.expand_dims(reward, 1)).float().to(device)
        )
        self.masks[step].copy_(mask)
        self.action_masks[step + 1].copy_(
            torch.from_numpy(action_masks).to(self.action_masks.device)
        )

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = (
                self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]
            )


def worker(remote, parent_remote, env: BotBowlWrapper, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = botbowl.make_bot("random")

    ppcg_wrapper: Optional[PPCGWrapper] = env.get_wrapper_with_type(PPCGWrapper)

    while True:
        command, data = remote.recv()
        if command == "step":
            steps += 1
            action, dif = data[0], data[1]
            if ppcg_wrapper is not None:
                ppcg_wrapper.difficulty = dif

            (spatial_obs, non_spatial_obs, action_mask), reward, done, info = env.step(
                action
            )

            game = env.game
            tds_scored = game.state.home_team.state.score - tds
            tds_opp_scored = game.state.away_team.state.score - tds_opp
            tds = game.state.home_team.state.score
            tds_opp = game.state.away_team.state.score

            if done or steps >= reset_steps:
                # If we get stuck or something - reset the environment
                if steps >= reset_steps:
                    print(
                        "Max. number of steps exceeded! Consider increasing the number."
                    )
                done = True
                env.root_env.away_agent = next_opp
                spatial_obs, non_spatial_obs, action_mask = env.reset()
                steps = 0
                tds = 0
                tds_opp = 0
            remote.send(
                (
                    spatial_obs,
                    non_spatial_obs,
                    action_mask,
                    reward,
                    tds_scored,
                    tds_opp_scored,
                    done,
                )
            )

        elif command == "reset":
            steps = 0
            tds = 0
            tds_opp = 0
            env.root_env.away_agent = next_opp
            spatial_obs, non_spatial_obs, action_mask = env.reset()
            remote.send((spatial_obs, non_spatial_obs, action_mask, 0.0, 0, 0, False))

        elif command == "swap":
            next_opp = data
        elif command == "close":
            break


class VecEnv:
    def __init__(self, envs):
        """
        envs: list of botbowl environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [
            Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)
        ]

        for p in self.ps:
            p.daemon = (
                True  # If the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment, returns the results as stacked numpy arrays
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(("reset", difficulty))
        results = [remote.recv() for remote in self.remotes]
        return tuple(map(np.stack, zip(*results)))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(("swap", agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def main():
    envs = VecEnv([make_env() for _ in range(num_processes)])

    env = make_env()
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    del (
        env,
        non_spat_obs,
        action_mask,
    )  # remove from scope to avoid confusion further down

    # MODEL
    ac_agent = CNNPolicy(
        spatial_obs_space,
        non_spatial_obs_space,
        hidden_nodes=num_hidden_nodes,
        kernels=num_cnn_kernels,
        residual_blocks=num_residual_blocks,
        actions=action_space,
    )
    ac_agent.to(DEVICE)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(
        steps_per_update,
        num_processes,
        spatial_obs_space,
        (1, non_spatial_obs_space),
        action_space,
    )
    if DEVICE.type == "cuda":
        memory.cuda()

    print(f"Training on device: {DEVICE}")

    # PPCG
    difficulty = 0.0 if ppcg else 1.0
    dif_delta = 0.01

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_tds = np.zeros(num_processes)
    proc_tds_opp = np.zeros(num_processes)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_td_rate_opp = []
    log_mean_reward = []
    log_difficulty = []

    checkpoint_scheduler = CheckpointOpponentScheduler(
        envs=envs,
        exp_identifier=run_id,
        make_agent_fn=make_agent_from_model,
        model_dir=model_dir,
        min_updates=min_updates_before_checkpoint,
        difficulty_threshold=difficulty_threshold,
        win_rate_threshold=win_rate_threshold,
    )
    plot_suffix = "checkpoint"

    # Reset environments
    (
        spatial_obs_np,
        non_spatial_obs_np,
        action_masks_np,
        *_,
    ) = envs.reset(difficulty)

    # Add first obs to memory
    spatial_obs = torch.from_numpy(spatial_obs_np).float().to(DEVICE)
    non_spatial_obs = torch.from_numpy(non_spatial_obs_np).float()
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1).to(DEVICE)
    action_masks = torch.from_numpy(action_masks_np).to(DEVICE)
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_masks)

    while all_steps < num_steps:
        for step in range(steps_per_update):
            _, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(memory.action_masks[step]),
            )

            actions_cpu = actions.detach().cpu()
            action_objects = (action[0] for action in actions_cpu.numpy())

            (
                spatial_obs,
                non_spatial_obs,
                action_masks,
                shaped_reward,
                tds_scored,
                tds_opp_scored,
                done,
            ) = envs.step(action_objects, difficulty=difficulty)

            proc_rewards += shaped_reward
            proc_tds += tds_scored
            proc_tds_opp += tds_opp_scored
            episodes += done.sum()

            # If done then clean the history of observations.
            for i in range(num_processes):
                if done[i]:
                    if proc_tds[i] > proc_tds_opp[i]:  # Win
                        wins.append(1)
                        difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:  # Loss
                        wins.append(0)
                        difficulty -= dif_delta
                    else:  # Draw
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ppcg:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0
                    proc_tds_opp[i] = 0

            # insert the step taken into memory
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=DEVICE,
            )

            memory.insert(
                step,
                spatial_obs,
                non_spatial_obs,
                actions.detach(),
                shaped_reward,
                masks,
                action_masks,
            )

        # -- TRAINING -- #

        # bootstrap next value
        next_value = ac_agent(
            Variable(memory.spatial_obs[-1], requires_grad=False),
            Variable(memory.non_spatial_obs[-1], requires_grad=False),
        )[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(memory.actions.view(-1, 1))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(
            spatial, non_spatial, actions, actions_mask
        )

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values

        # Normalizacja Do stestowania czy się nie wywali
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss = advantages.pow(2).mean()
        # value_losses.append(value_loss)

        # Compute loss
        action_loss = -(advantages * action_log_probs).mean()
        # policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (
            value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef
        )
        # dodane z sugestii GPT
        total_loss = total_loss / num_processes

        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])
        memory.action_masks[0].copy_(memory.action_masks[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()

            swapped_path = checkpoint_scheduler.maybe_swap(
                all_updates, ac_agent, difficulty, win_rate
            )
            if swapped_path is not None:
                print(
                    f"[CheckpointScheduler] Swapped opponent to {swapped_path} "
                    f"(difficulty={difficulty:.2f}, win_rate={win_rate:.2%})"
                )

            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, TD rate: {:.2f}, TD rate opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}".format(
                all_updates,
                all_episodes,
                all_steps,
                win_rate,
                td_rate,
                td_rate_opp,
                mean_reward,
                difficulty,
            )

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n".format(
                all_updates,
                all_episodes,
                all_steps,
                win_rate,
                td_rate,
                td_rate_opp,
                mean_reward,
                difficulty,
            )

            # Save to files
            log_path = os.path.join(log_dir, f"{run_id}.dat")
            print(f"Save log to {log_path}")
            with open(log_path, "a") as myfile:
                myfile.write(log_to_file)

            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # plot
            n = 3
            if ppcg:
                n += 1
            fig, axs = plt.subplots(1, n, figsize=(4 * n, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title("Reward")
            # axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axs[1].plot(log_steps, log_td_rate, label="Learner")
            axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
            axs[1].set_title("TD/Episode")
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            axs[1].legend()
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title("Win rate")
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title("Difficulty")
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            plot_name = f"{run_id}_{plot_suffix}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            fig.savefig(plot_path)
            plt.close("all")

        if save_interval > 0 and all_updates % save_interval == 0:
            model_name = f"{MODEL_KIND}_{timestamp_now()}_upd{all_updates}.nn"
            model_path = os.path.join(model_dir, model_name)
            torch.save(ac_agent, model_path)
            print(f"Saved model checkpoint to {model_path}")

    model_name = f"{MODEL_KIND}_{timestamp_now()}_final.nn"
    model_path = os.path.join(model_dir, model_name)
    torch.save(ac_agent, model_path)
    envs.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
