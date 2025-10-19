from functools import partial
from multiprocessing import Process, Pipe
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple, Iterable

import botbowl

from botbowl_update.a2c_env import A2C_Reward
from botbowl.ai.env import (
    BotBowlEnv,
    RewardWrapper,
    EnvConf,
    ScriptedActionWrapper,
    BotBowlWrapper,
    PPCGWrapper,
)

# Finding GPU
print("Cuda available: " + str(torch.cuda.is_available()))
found_device = torch.cuda.current_device()
device = torch.device(found_device if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))

# number of cpu processes to use in training
num_processes = 2
# changing size of field according to gathered points
ppcg = False
# number of steps taken to reset?
reset_steps = 5000


# specifing environment
env_size = 11  # pełno rozmiarowe boisko i pełne drużyny
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=False)


def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    env = RewardWrapper(
        env, home_reward_func=A2C_Reward(), away_reward_func=A2C_Reward()
    )


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


def main():
    # stworz srodowiska
    envs = VecEnv([make_env() for _ in range(num_processes)])
    # do utworzenia zmiennych odpowiednich rozmiarowe
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

    # stworz modele
    # ucz modele
    # z random bot
    # z samymi soba
    # z innymi modelami
    # zapisuj wyniki


if __name__ == "__main__":
    main()
