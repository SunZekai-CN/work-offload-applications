import argparse
import os
import pickle
import shutil
import sys
import random
import torch

sys.path.append(".")

import time
from collections import deque
from itertools import chain
from sys import getsizeof as getsize

from core.utils.trainer_utils import setup_agent
from core.env import VecEnv
from core.experiences import ExperienceManager

from sys import getsizeof, stderr
from itertools import chain
from collections import deque
#
# try:
#     from reprlib import repr
# except ImportError:
#     pass
#
#
# def total_size(o, handlers={}, verbose=False):
#     """ Returns the approximate memory footprint an object and all of its contents.
#     Automatically finds the contents of the following builtin containers and
#     their subclasses:  tuple, list, deque, dict, set and frozenset.
#     To search other containers, add handlers to iterate over their contents:
#         handlers = {SomeContainerClass: iter,
#                     OtherContainerClass: OtherContainerClass.get_elements}
#     """
#     dict_handler = lambda d: chain.from_iterable(d.items())
#     all_handlers = {tuple: iter,
#                     list: iter,
#                     deque: iter,
#                     dict: dict_handler,
#                     set: iter,
#                     frozenset: iter,
#                     }
#     all_handlers.update(handlers)  # user handlers take precedence
#     seen = set()  # track which object id's have already been seen
#     default_size = getsizeof(0)  # estimate sizeof object without __sizeof__
#
#     def sizeof(o):
#         if id(o) in seen:  # do not double count the same object
#             return 0
#         seen.add(id(o))
#         s = getsizeof(o, default_size)
#
#         if verbose:
#             print(s, type(o), repr(o), file=stderr)
#
#         for typ, handler in all_handlers.items():
#             if isinstance(o, typ):
#                 s += sum(map(sizeof, handler(o)))
#                 break
#         return s
#
#     return sizeof(o)
#


def eval_agent(data=None, name=None, checkpoint=None, n_envs=None, n_evals=None, split=None,device = None, **config):
    env_config, meta, handler, trainer, agent, init_env = \
        setup_agent(data=data, checkpoint=checkpoint, **config)
    total_input_size = total_running_time = total_inf_time = 0
    idx = list(range(n_evals))
    random.shuffle(idx)
    for i in range(n_evals):
        running_start_time = time.time()
        file_path = f"GPUoffload_test/saved_obs/{idx[i]}/histories.pkl"
        with open(file_path, 'rb') as file:
            histories = pickle.load(file)
            total_input_size += os.stat(file_path).st_size
        file_path = f"GPUoffload_test/saved_obs/{idx[i]}/new_episodes.pkl"
        with open(file_path, 'rb') as file:
            new_episodes = pickle.load(file)
            total_input_size += os.stat(file_path).st_size
        inf_start_time = time.time()
        _, _, _ = agent.policy(histories, new_episodes, None)
        end_time = time.time()
        
        print(f"processing {i + 1} / {n_evals}")
        total_inf_time += (end_time - inf_start_time)
        total_running_time += (end_time - running_start_time)
        print(f"T_robot : {(end_time - inf_start_time)*1000:.2f} ms, average :{total_inf_time/(i+1)*1000:.2f} ms (GPU computation time on robot)")
        print(f"Service time : {(end_time - running_start_time)*1000:.2f} ms, average :{total_running_time/(i+1)*1000:.2f} ms")



def add_eval_args(parser):
    parser.add_argument("checkpoint", help="checkpoint")
    parser.add_argument("--data", default=None, help="path to data directory")
    parser.add_argument("--name", default="eval", help="name of save directory")
    parser.add_argument("--split", default="val", help="train / val split")
    parser.add_argument("--device", "-d", default="cuda", help="device (cuda, cuda:0, cpu, etc.)")
    parser.add_argument("--n_evals", "-times", type=int, help="number of evaluation runs", default=100)
    parser.add_argument("--n_envs", "-env", type=int, help="number of envs", default=3)
    return parser


def main():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    add_eval_args(parser)
    config = vars(parser.parse_args())

    checkpoint = config['checkpoint']
    if not config['data']:
        config['data'] = os.path.join(os.path.dirname(checkpoint), "..", "..", "..")
    eval_agent(**config)

if __name__ == "__main__":
    main()
