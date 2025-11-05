from multiprocessing import Process, Pipe
import numpy as np
import torch
from typing import Tuple, Iterable, Optional

import botbowl

from a2c_env import A2C_Reward, a2c_scripted_actions
from botbowl.ai.env import (
    BotBowlEnv,
    RewardWrapper,
    EnvConf,
    BotBowlWrapper,
    PPCGWrapper,
    ScriptedActionWrapper,
)

from my_gpu_agents import SpatialInceptionCNN

# Finding GPU
print("Cuda available: " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device: " + str(device))

# number of cpu processes to use in training
num_processes = 8
# changing size of field according to gathered points
ppcg = False
# number of steps taken to reset?
reset_steps = 5000

# Training hyper-parameters
NUM_STEPS = 5000
STEPS_PER_UPDATE = 16
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.05
LEARNING_RATE = 1e-3
LOG_INTERVAL = 10

# specifing environment
env_size = 7  # pełno rozmiarowe boisko i pełne drużyny
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)


def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    env = ScriptedActionWrapper(env, scripted_func=a2c_scripted_actions)
    return env


# Ta funkcja steruje grą
# Albo ja wyeliminować albo przerobić tak, aby z wybranych modeli brala przeciwnikow
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


# rozsyla polecenia do workerow z innych procesow
class VecEnv:
    def __init__(self, envs):
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [
            Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)
        ]

        for p in self.ps:
            p.daemon = True  # If the main process crashes we still going
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each env, return the results as stacked nuympy arrays
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


def _ensure_numpy(array, dtype):
    if isinstance(array, np.ndarray):
        if array.dtype == dtype:
            return np.ascontiguousarray(array)
        if array.dtype == object:
            stacked = [_ensure_numpy(a, dtype) for a in array]
            array = np.stack(stacked, axis=0)
            return np.ascontiguousarray(array.astype(dtype, copy=False))
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    if isinstance(array, (list, tuple)):
        stacked = [_ensure_numpy(a, dtype) for a in array]
        array = np.stack(stacked, axis=0)
        return np.ascontiguousarray(array.astype(dtype, copy=False))

    return np.ascontiguousarray(np.asarray(array, dtype=dtype))


def _to_device_tensor(array, np_dtype, torch_dtype, device):
    np_array = _ensure_numpy(array, np_dtype)
    return torch.as_tensor(np_array, dtype=torch_dtype, device=device)


def _prepare_action_mask(mask: np.ndarray) -> np.ndarray:
    mask_np = _ensure_numpy(mask, np.bool_)
    if mask_np.size == 0:
        return np.ones_like(mask_np, dtype=np.bool_)
    if not mask_np.any():
        mask_np = np.ones_like(mask_np, dtype=np.bool_)
    return mask_np


def _prepare_action_mask_batch(mask_batch: np.ndarray) -> np.ndarray:
    if mask_batch.ndim == 1:
        return _prepare_action_mask(mask_batch)
    prepared = [_prepare_action_mask(mask) for mask in mask_batch]
    return np.stack(prepared, axis=0)


def _ensure_valid_mask_tensor(mask_tensor: torch.Tensor) -> torch.Tensor:
    mask_tensor = mask_tensor.to(torch.bool)
    if mask_tensor.ndim == 1:
        if not mask_tensor.any():
            mask_tensor = torch.ones_like(mask_tensor, dtype=torch.bool)
        return mask_tensor

    flat = mask_tensor.view(mask_tensor.shape[0], -1)
    invalid_rows = ~flat.any(dim=1)
    if invalid_rows.any():
        mask_tensor[invalid_rows] = True
    return mask_tensor


class Memory:
    def __init__(
        self,
        steps_per_update: int,
        num_envs: int,
        spatial_shape: Tuple[int, ...],
        non_spatial_dim: int,
        action_space: int,
        device: torch.device,
    ):
        self.device = device
        self.steps_per_update = steps_per_update
        self.num_envs = num_envs
        self.spatial_obs = torch.zeros(
            steps_per_update + 1,
            num_envs,
            *spatial_shape,
            dtype=torch.float32,
            device=device,
        )
        self.non_spatial_obs = torch.zeros(
            steps_per_update + 1,
            num_envs,
            non_spatial_dim,
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(
            steps_per_update,
            num_envs,
            1,
            dtype=torch.float32,
            device=device,
        )
        self.masks = torch.ones(
            steps_per_update + 1,
            num_envs,
            1,
            dtype=torch.float32,
            device=device,
        )
        self.returns = torch.zeros(
            steps_per_update + 1,
            num_envs,
            1,
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            steps_per_update,
            num_envs,
            1,
            dtype=torch.long,
            device=device,
        )
        self.action_masks = torch.zeros(
            steps_per_update + 1,
            num_envs,
            action_space,
            dtype=torch.bool,
            device=device,
        )

    def set_initial(
        self,
        spatial_obs: np.ndarray,
        non_spatial_obs: np.ndarray,
        action_mask: np.ndarray,
    ):
        self.spatial_obs[0].copy_(
            _to_device_tensor(spatial_obs, np.float32, torch.float32, self.device)
        )
        self.non_spatial_obs[0].copy_(
            _to_device_tensor(non_spatial_obs, np.float32, torch.float32, self.device)
        )
        self.action_masks[0].copy_(
            _to_device_tensor(action_mask, np.bool_, torch.bool, self.device)
        )

    def insert(
        self,
        step: int,
        spatial_obs: np.ndarray,
        non_spatial_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        action_masks: np.ndarray,
    ):
        self.spatial_obs[step + 1].copy_(
            _to_device_tensor(spatial_obs, np.float32, torch.float32, self.device)
        )
        self.non_spatial_obs[step + 1].copy_(
            _to_device_tensor(non_spatial_obs, np.float32, torch.float32, self.device)
        )
        actions_tensor = _to_device_tensor(actions, np.int64, torch.long, self.device)
        if actions_tensor.ndim == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)
        self.actions[step].copy_(actions_tensor)
        self.rewards[step].copy_(
            _to_device_tensor(rewards, np.float32, torch.float32, self.device).unsqueeze(-1)
        )
        self.masks[step].copy_(
            _to_device_tensor(masks, np.float32, torch.float32, self.device)
        )
        self.action_masks[step + 1].copy_(
            _to_device_tensor(action_masks, np.bool_, torch.bool, self.device)
        )

    def compute_returns(self, next_value: torch.Tensor, gamma: float) -> torch.Tensor:
        if next_value.ndim == 1:
            next_value = next_value.unsqueeze(-1)
        self.returns[-1].copy_(next_value)
        for step in reversed(range(self.steps_per_update)):
            self.returns[step] = (
                self.rewards[step] + gamma * self.masks[step] * self.returns[step + 1]
            )
        return self.returns


def main():
    torch.manual_seed(0)

    env = make_env()
    print(env)
    spat_obs, non_spat_obs, action_mask = env.reset()
    action_mask = _prepare_action_mask(action_mask)
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_dim = non_spat_obs.shape[0]
    action_space = len(action_mask)
    env.close()

    num_hidden_nodes = 128
    num_residual_blocks = 2
    num_cnn_kernels = [(18, 3), (18, 5), (8, 7)]
    model = SpatialInceptionCNN(
        spatial_obs_space,
        non_spatial_obs_dim,
        num_hidden_nodes,
        num_cnn_kernels,
        num_residual_blocks,
        action_space,
    ).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    envs = VecEnv([make_env() for _ in range(num_processes)])

    difficulty = 0.0 if ppcg else 1.0
    dif_delta = 0.01 if ppcg else 0.0

    memory = Memory(
        STEPS_PER_UPDATE,
        num_processes,
        spatial_obs_space,
        non_spatial_obs_dim,
        action_space,
        device,
    )

    (
        spatial_obs,
        non_spatial_obs,
        action_masks,
        shaped_reward,
        tds_scored,
        tds_opp_scored,
        done,
    ) = envs.reset(difficulty)

    action_masks = _prepare_action_mask_batch(action_masks)
    memory.set_initial(spatial_obs, non_spatial_obs, action_masks)

    proc_rewards = np.zeros(num_processes, dtype=np.float32)
    proc_tds = np.zeros(num_processes, dtype=np.float32)
    proc_tds_opp = np.zeros(num_processes, dtype=np.float32)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []

    update_idx = 0
    total_steps = 0
    total_episodes = 0
    episodes = 0

    while total_steps < NUM_STEPS:
        for step in range(STEPS_PER_UPDATE):
            spatial_tensor = memory.spatial_obs[step]
            non_spatial_tensor = memory.non_spatial_obs[step]
            action_mask_tensor = memory.action_masks[step]

            _, action_probs = model.get_action_probs(
                spatial_tensor, non_spatial_tensor, action_mask_tensor
            )
            dist = torch.distributions.Categorical(probs=action_probs)
            actions = dist.sample()

            action_list = actions.view(-1).tolist()
            (
                next_spatial_obs,
                next_non_spatial_obs,
                next_action_masks,
                shaped_reward,
                tds_scored,
                tds_opp_scored,
                done,
            ) = envs.step(action_list, difficulty=difficulty)

            next_action_masks = _prepare_action_mask_batch(next_action_masks)

            proc_rewards += shaped_reward
            proc_tds += tds_scored
            proc_tds_opp += tds_opp_scored
            episodes += done.sum()

            for i in range(num_processes):
                if done[i]:
                    if proc_tds[i] > proc_tds_opp[i]:
                        wins.append(1.0)
                        difficulty += dif_delta
                    elif proc_tds[i] < proc_tds_opp[i]:
                        wins.append(0.0)
                        difficulty -= dif_delta
                    else:
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ppcg:
                        difficulty = min(1.0, max(0.0, difficulty))
                    else:
                        difficulty = 1.0
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0.0
                    proc_tds[i] = 0.0
                    proc_tds_opp[i] = 0.0

            masks = np.array([[0.0] if done_ else [1.0] for done_ in done], dtype=np.float32)

            memory.insert(
                step,
                next_spatial_obs,
                next_non_spatial_obs,
                np.array(action_list, dtype=np.int64),
                shaped_reward,
                masks,
                next_action_masks,
            )

        with torch.no_grad():
            next_value, _ = model(
                memory.spatial_obs[-1], memory.non_spatial_obs[-1]
            )

        memory.compute_returns(next_value.detach(), GAMMA)

        spatial_batch = memory.spatial_obs[:-1].view(-1, *spatial_obs_space)
        non_spatial_batch = memory.non_spatial_obs[:-1].view(-1, non_spatial_obs_dim)
        actions_batch = memory.actions.view(-1, 1)
        action_masks_batch = memory.action_masks[:-1].view(-1, action_space)
        returns_batch = memory.returns[:-1].view(-1, 1)

        action_log_probs, values, dist_entropy = model.evaluate_actions(
            spatial_batch, non_spatial_batch, actions_batch, action_masks_batch
        )

        advantages = returns_batch - values
        value_loss = advantages.pow(2).mean()
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        loss = value_loss * VALUE_LOSS_COEF + policy_loss - dist_entropy * ENTROPY_COEF

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])
        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.action_masks[0].copy_(memory.action_masks[-1])

        update_idx += 1
        total_steps += num_processes * STEPS_PER_UPDATE
        total_episodes += episodes
        episodes = 0

        if update_idx % LOG_INTERVAL == 0:
            avg_return = returns_batch.mean().item()
            print(
                f"Update {update_idx}: total_steps={total_steps}, "
                f"loss={loss.item():.3f}, value_loss={value_loss.item():.3f}, "
                f"policy_loss={policy_loss.item():.3f}, avg_return={avg_return:.3f}, "
                f"episodes={total_episodes}"
            )

            if len(episode_rewards) >= num_processes:
                mean_reward = float(np.mean(episode_rewards))
                td_rate = float(np.mean(episode_tds))
                td_rate_opp = float(np.mean(episode_tds_opp))
                win_rate = float(np.mean(wins)) if wins else 0.0
                print(
                    f"Stats -> Reward: {mean_reward:.3f}, Win rate: {win_rate:.2f}, "
                    f"TD rate: {td_rate:.2f}, TD rate opp: {td_rate_opp:.2f}, "
                    f"Difficulty: {difficulty:.2f}"
                )
                episode_rewards.clear()
                episode_tds.clear()
                episode_tds_opp.clear()
                wins.clear()

        if total_steps >= NUM_STEPS:
            break

    envs.close()


if __name__ == "__main__":
    main()
