#!/usr/bin/env python3

import gymnasium as gym
from torch.autograd import Variable
from saving_model import save_model, load_model
import numpy as np
import torch
import os
import uuid
from multiprocessing import Process, Pipe
from typing import Tuple, Iterable, Optional
import botbowl

# środowisko z rendererem
from botbowl.ai.env_render import EnvRenderer

# Czy bezpośrednio środowisko BotBowlEnv : czyste env
#                        czy BotBowlWrapper : uproszczony interface BotBowlEnv
#                            RewardWrapper : Wraper do BBW z zliczeniem nagrody
#                            PPCGWrapper : Wrapper do BBW z zmiennym rozmiarem obszaru punktującego.
# w przykladzie dla a2c byly wszystkie ziamportowane
# Dlaczego?
from botbowl.ai.env import (
    BotBowlEnv,
    RewardWrapper,
    EnvConf,
    ScriptedActionWrapper,
    BotBowlWrapper,
    PPCGWrapper,
)

# nagroda dla sieci
from examples.a2c.a2c_env import A2C_Reward


# przygotowanie GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Czy zmienny rozmiar strefy punktowania
ppcg = False
# Agenci

agents = []


# Environment - tworzenie Środowisk


env_size = 11  # Options are 1,3,5,7,11
env_name = f"botbowl-{env_size}"
env_conf = EnvConf(size=env_size, pathfinding=True)


# funkcja do tworzenia środowiska
def make_env():
    env = BotBowlEnv(env_conf)
    if ppcg:
        env = PPCGWrapper(env)
    # TO-DO Co się stanie jak będę chciał też uczyć drużynę gości
    env = RewardWrapper(env, home_reward_func=A2C_Reward())
    return env


# Training configuration.
# TO-DO Musze ogarnac co oznaczaja wartosci od gamma do max_grad_norm
num_steps = 1000000
num_processes = 8
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 50
save_interval = 10
ppcg = True
reset_steps = 200


# Podprocesy odpowiedzialne za nauke modelu
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


# wektor środowisk
#
# zamiast wysyłać pojedyńczo step wolałbym, aby procesy rozgrywały gry jak najszybciej do kończa.
class VecEnv:
    def __init__(self, envs):
        """
        envs: Lista środowisk botbowl które mają działać równolegle
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [
            Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def step(self, actions: Iterable[int], difficulty=1.0) -> Tuple[np.ndarray, ...]:
        """
        Takes one step in each environment:: Czy napewno chcę to w ten sposób zrobić.
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


class Memory(object):
    def __init__(
        self,
        steps_per_update,
        num_processes,
        spatial_obs_shape,
        non_spatial_obs_shape,
        action_space,
        device,
    ):
        self.device = device
        self.spatial_obs = torch.zeros(
            steps_per_update + 1, num_processes, *spatial_obs_shape, device=device
        )
        self.non_spatial_obs = torch.zeros(
            steps_per_update + 1, num_processes, *non_spatial_obs_shape, device=device
        )
        self.rewards = torch.zeros(steps_per_update, num_processes, 1, device=device)
        self.returns = torch.zeros(
            steps_per_update + 1, num_processes, 1, device=device
        )
        action_shape = 1
        self.actions = torch.zeros(
            steps_per_update,
            num_processes,
            action_shape,
            dtype=torch.long,
            device=device,
        )
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1, device=device)
        self.action_masks = torch.zeros(
            steps_per_update + 1,
            num_processes,
            action_space,
            dtype=torch.bool,
            device=device,
        )

    def to(self, device: torch.device):
        self.device = device
        self.spatial_obs = self.spatial_obs.to(device)
        self.non_spatial_obs = self.non_spatial_obs.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.action_masks = self.action_masks.to(device)
        return self

    def cuda(self):
        return self.to(torch.device("cuda"))

    def insert(
        self, step, spatial_obs, non_spatial_obs, action, reward, mask, action_masks
    ):
        spatial_tensor = torch.from_numpy(spatial_obs).float().to(self.device)
        non_spatial_tensor = (
            torch.from_numpy(np.expand_dims(non_spatial_obs, axis=1))
            .float()
            .to(self.device)
        )
        action_tensor = action.to(self.device).long()
        reward_tensor = (
            torch.from_numpy(np.expand_dims(reward, 1)).float().to(self.device)
        )
        mask_tensor = mask.to(self.device)
        action_mask_tensor = torch.from_numpy(action_masks).to(self.device)

        self.spatial_obs[step + 1].copy_(spatial_tensor)
        self.non_spatial_obs[step + 1].copy_(non_spatial_tensor)
        self.actions[step].copy_(action_tensor)
        self.rewards[step].copy_(reward_tensor)
        self.masks[step].copy_(mask_tensor)
        self.action_masks[step + 1].copy_(action_mask_tensor)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.shape[0])):
            self.returns[step] = (
                self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]
            )


# TO-DO
def main():
    process_number = 8
    envs = VecEnv([make_env() for _ in range(process_number)])

    # czy trening odbędzie się na GPU?
    # to środowisko jest rejestrowane w botbowl ai __init__ klasa env bez UI i env_rendere z wygladem
    # w tej chwili chce bez renderu
    # Work rmeotest trafia do procesow srodowisk, a remotes zostaje w procesie glownym

    env = make_env()
    spat_obs, non_spat_obs, action_mask = env.reset()
    spatial_obs_space = spat_obs.shape
    non_spatial_obs_space = non_spat_obs.shape[0]
    action_space = len(action_mask)
    # jakieś tricki. Po co to usuwaja
    # chyba rozumiem. Chcą same rozmiary, aby stworzyć agents
    del (env, non_spat_obs, action_mask)

    # Tutaj mogę tworzyć swoje model
    # MODEL
    ac_agent = CNNPolicy(
        spatial_obs_space,
        non_spatial_obs_space,
        hidden_nodes=num_hidden_nodes,  # to bierze z zewnątrz
        kernels=num_cnn_kernels,  # to też z zewnątrz
        actions=action_space,
    ).to(device)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(
        steps_per_update,
        num_processes,
        spatial_obs_space,
        (1, non_spatial_obs_space),
        action_space,
        device=device,
    )

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

    # self-play
    selfplay_next_save = selfplay_save_steps
    selfplay_next_swap = selfplay_swap_steps
    selfplay_models = 0

    if selfplay:
        model_name = f"{exp_id}_selfplay_0.nn"
        model_path = os.path.join(model_dir, model_name)
        torch.save(ac_agent, model_path)
        self_play_agent = make_agent_from_model(name=model_name, filename=model_path)
        envs.swap(self_play_agent)
        selfplay_models += 1

    # Reset environments / Chyba nalezy to zrobić na początku symulacji, nawet jeśli jest to świerze środowisko.
    reset_outputs = envs.reset(difficulty)
    spatial_obs_np, non_spatial_obs_np, action_masks_np = reset_outputs[:3]
    spatial_obs = torch.from_numpy(spatial_obs_np).float().to(device)
    non_spatial_obs = torch.from_numpy(non_spatial_obs_np).float().to(device)
    action_masks = torch.from_numpy(action_masks_np).to(device)

    # Add first obserwation to memory
    non_spatial_obs = torch.unsqueeze(non_spatial_obs, dim=1)
    memory.spatial_obs[0].copy_(spatial_obs)
    # co robi "copy_" ?
    # Kopiuje wartości modyfikując wartości wywołującego obiektu,
    # a nie tworząc nowy lub zmieniając referencje
    memory.non_spatial_obs[0].copy_(non_spatial_obs)
    memory.action_masks[0].copy_(action_mask)

    while all_steps < num_steps:
        for step in range(steps_per_update):
            _, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(memory.action_masks[step]),
            )

            actions_cpu = actions.detach().cpu()
            action_objects = (action[0] for action in actions_cpu.numpy())

        # Chyba tutaj opróczprzekazania możliwych akcji zapisujemy wynik.
        (
            spatial_obs,
            non_spatial_obs,
            action_masks,
            shaped_reward,
            tds_scored,
            tds_opp_scored,
            done,
        ) = envs.step(action_objects, difficulty=difficulty)

        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )
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
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        memory.insert(
            step,
            spatial_obs,
            non_spatial_obs,
            actions.data,
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
    value_loss = advantages.pow(2).mean()
    # value_losses.append(value_loss)

    # Compute loss
    action_loss = -(Variable(advantages.data) * action_log_probs).mean()
    # policy_losses.append(action_loss)

    optimizer.zero_grad()

    total_loss = (
        value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef
    )
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

    # Self-play save
    if selfplay and all_steps >= selfplay_next_save:
        selfplay_next_save = max(
            all_steps + 1, selfplay_next_save + selfplay_save_steps
        )
        model_name = f"{exp_id}_selfplay_{selfplay_models}.nn"
        model_path = os.path.join(model_dir, model_name)
        print(f"Saving {model_path}")
        torch.save(ac_agent, model_path)
        selfplay_models += 1

    # Self-play swap
    if selfplay and all_steps >= selfplay_next_swap:
        selfplay_next_swap = max(
            all_steps + 1, selfplay_next_swap + selfplay_swap_steps
        )
        lower = max(0, selfplay_models - 1 - (selfplay_window - 1))
        i = random.randint(lower, selfplay_models - 1)
        model_name = f"{exp_id}_selfplay_{i}.nn"
        model_path = os.path.join(model_dir, model_name)
        print(f"Swapping opponent to {model_path}")
        envs.swap(make_agent_from_model(name=model_name, filename=model_path))

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
        log_path = os.path.join(log_dir, f"{exp_id}.dat")
        print(f"Save log to {log_path}")
        with open(log_path, "a") as myfile:
            myfile.write(log_to_file)

        print(log)

        episodes = 0
        value_losses.clear()
        policy_losses.clear()

        # Save model
        model_name = f"{exp_id}.nn"
        model_path = os.path.join(model_dir, model_name)
        torch.save(ac_agent, model_path)

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
        axs[1].set_title("TD/Episode")
        axs[1].set_ylim(bottom=0.0)
        axs[1].set_xlim(left=0)
        if selfplay:
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
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
        plot_name = f"{exp_id}_{'_selfplay' if selfplay else ''}.png"
        plot_path = os.path.join(plot_dir, plot_name)
        fig.savefig(plot_path)
        plt.close("all")

    model_name = f"{exp_id}.nn"
    model_path = os.path.join(model_dir, model_name)
    torch.save(ac_agent, model_path)
    envs.close()


if __name__ == "__main__":
    main()
