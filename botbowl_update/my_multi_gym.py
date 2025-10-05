#!/usr/bin/env python3

import gymnasium as gym
from saving_model import save_model, load_model
import numpy as np
from multiprocessing import Process, Pipe

# środowisko z rendererem
from botbowl.ai.env_render import EnvRenderer

# Czy bezpośrednio środowisko BotBowlEnv
#                        czy BotBowlWrapper
#                            RewardWrapper
#                            PPCGWrapper
from botbowl.ai.env import BotBowlWrapper


def main():
    # liczba środowisk nenvs, ale chyba lepiej uzależnić ją od ilości trenowanych modeli
    nenvs = 4
    envs = []
    renderers = []

    # czy trening odbędzie się na GPU?
    for _ in range(nenvs):
        # to środowisko jest rejestrowane w botbowl ai __init__ klasa env bez UI i env_rendere z wygladem
        env = gym.make("botbowl-11-v4")
        envs.append(env)
        # w tej chwili chce bez renderu
        # renderers.append(EnvRenderer(env))
    # tworzy kanaly komunkacji  miedzy soba a środowiskami
    # wynik operacji [(a1, a2, a3, a4), (b1, b2, b3, b4)] gdzie a i b to są konce Pipe
    # Work rmeotest trafia do procesow srodowisk, a remotes zostaje w procesie glownym
    remotes, work_remotes = zip(*[Pipe() for _ in range(nenvs)])

    ps = [
        Process(target=worker, args=(work_remote, remote, env))
        for (work_remote, remote, env) in zip(work_remotes, remotes, envs)
    ]

    for p in ps:
        p.daemon = (
            True  # If the main process crashes, we should not cause things to hang
        )
        p.start()

    for remote in work_remotes:
        remote.close()

    for i in range(1000):
        print(i)
        for remote in remotes:
            remote.send("step")
        received_games = [remote.recv() for remote in remotes]

        for game, renderer in zip(received_games, renderers):
            renderer.env.game = game
            renderer.render()

    for remote in remotes:
        remote.send("close")

    for p in ps:
        p.join()
