import argparse
import os
import pickle
import shutil
import sys
import time

sys.path.append(".")

from core.utils.trainer_utils import setup_agent
from core.env import VecEnv
from core.experiences import ExperienceManager


def eval_agent(data=None, name=None, checkpoint=None, n_envs=None, n_evals=None, split=None, **config):
    env_config, meta, handler, trainer, agent, init_env = \
        setup_agent(data=data, checkpoint=checkpoint, **config)
    total_input_size = total_running_time = total_inf_time = 0
    for i in range(n_evals):
        running_start_time =time.time()
        with open(f"GPUoffload_test/saved_obs/{i - i % 1000}/histories.pkl", 'rb') as file:
            histories = pickle.load(file)
        with open(f"GPUoffload_test/saved_obs/{i - i % 1000}/new_episodes.pkl", 'rb') as file:
            new_episodes = pickle.load(file)
        total_input_size += (sys.getsizeof(histories) + sys.getsizeof(new_episodes))
        inf_start_time = time.time()
        _, _, _ = agent.policy(histories, new_episodes, None)
        end_time = time.time()
        total_inf_time += (end_time - inf_start_time)
        total_running_time+=(end_time - running_start_time)
        print(f"processing {i + 1} / {n_evals}")

    print(f"Average input size: {total_input_size / n_evals} byte,",
          f"Average running time: {total_running_time/i}, "
          f"Average inference time: {total_inf_time / n_evals}")


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
