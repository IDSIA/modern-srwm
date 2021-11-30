# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import multiprocessing as mp
import threading
import time

import numpy as np
import libtorchbeast
from torchbeast_procgen import procgen_wrappers
import gym


# lexicographically sorted list
list_procgen_env_lex = [
    'bigfish',
    'bossfight',
    'caveflyer',
    'chaser',
    'climber',
    'coinrun',
    'dodgeball',
    'fruitbot',
    'heist',
    'jumper',
    'leaper',
    'maze',
    'miner',
    'ninja',
    'plunder',
    'starpilot',
]

# interesting ones first, we use this list
list_procgen_env = [
    'bigfish',
    'fruitbot',
    'maze',
    'leaper',
    'plunder',
    'starpilot',
    'miner',
    'bossfight',
    'caveflyer',
    'chaser',
    'climber',
    'coinrun',
    'dodgeball',
    'heist',
    'jumper',
    'ninja',
]

# env with memory mode extension
list_procgen_env_mem = [
    'dodgeball',
    'heist',
    'maze',
    'miner',
    'caveflyer',
    'jumper',
]


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Gym environment.')
parser.add_argument('--multi_env', default=1, type=int, metavar='N',
                    help='number of env to jointly train on.')
parser.add_argument('--allow_oov', action="store_true",
                    help='Allow action space larger than the env specific one.'
                    ' All out-of-vocab action will be mapped to NoOp.')
parser.add_argument('--num_levels', default=0, type=int, metavar='N',
                    help='Procgen num_levels.')
parser.add_argument('--start_level', default=0, type=int, metavar='N',
                    help='Procgen start_level.')
parser.add_argument('--distribution_mode', type=str, default='hard',
                    choices=[
                        'easy', 'hard', 'extreme', 'memory', 'exploration'],
                    help='distribution mode.')
# yapf: enable


class Env:
    def reset(self):
        print("reset called")
        return np.ones((4, 84, 84), dtype=np.uint8)

    def step(self, action):
        frame = np.zeros((4, 84, 84), dtype=np.uint8)
        return frame, 0.0, False, {}  # First three mandatory.


def create_env(env_name, num_levels=0, start_level=0, distribution_mode="hard",
               rand_seed=None, lock=threading.Lock()):
    with lock:  # Atari isn't threadsafe at construction time.
        return procgen_wrappers.wrap_pytorch(
            procgen_wrappers.wrap_deepmind(
                gym.make(env_name,
                         num_levels=num_levels,
                         start_level=start_level,
                         distribution_mode=distribution_mode,
                         rand_seed=rand_seed),
                clip_rewards=False,
            )
        )


def serve(env_name, num_levels, start_level, distribution_mode,
          server_address):
    init = Env if env_name == "Mock" else lambda: create_env(
        env_name, num_levels, start_level, distribution_mode)
    server = libtorchbeast.Server(init, server_address=server_address)
    server.run()


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception(
            "--pipes_basename has to be of the form unix:/some/path.")

    if flags.distribution_mode == 'memory':  # for multi_env training
        list_env = list_procgen_env_mem
    else:
        list_env = list_procgen_env

    processes = []
    for i in range(flags.num_servers):
        if flags.multi_env > 1:
            env_name = list_env[i % flags.multi_env]
            env_name = f"procgen:procgen-{env_name}-v0"
        else:
            env_name = flags.env
        print(f"Server {i} on {env_name}")
        # distributed mode and rand_seed left to default.
        p = mp.Process(
            target=serve, args=(
                env_name, flags.num_levels, flags.start_level,
                flags.distribution_mode, f"{flags.pipes_basename}.{i}"),
            daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    flags = parser.parse_args()
    print(f"Env: {flags.env}")
    main(flags)
