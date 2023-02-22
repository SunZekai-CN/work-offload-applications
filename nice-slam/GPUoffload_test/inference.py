import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
import sys

sys.path.append(".")
from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--times', type=int, default=100)
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    file_num = len([f for f in os.listdir(f"GPUoffload_test/saved_obs/")])
    assert file_num > 0
    n_steps = args.times
    slam = NICE_SLAM(cfg, args)
    mapper = slam.mapper
    total_input_size = total_running_time = total_inf_time = 0
    for i in range(n_steps):
        time_ckp_0 = time.time()
        arg_names = ["cur_gt_color", "cur_gt_depth", "gt_cur_c2w", "keyframe_dict", "keyframe_list", "cur_c2w"]
        arg_dict = {}
        device = next(slam.shared_decoders.parameters()).device
        # print("===================>device: ", device)
        for arg_name in arg_names:
            file_path = f"GPUoffload_test/saved_obs/{i - i % file_num}/{arg_name}.pkl"
            with open(file_path, 'rb') as file:
                arg_dict[arg_name] = pickle.load(file)
                if isinstance(arg_dict[arg_name],torch.Tensor):
                    arg_dict[arg_name].to(device)
                total_input_size += os.stat(file_path).st_size

        time_ckp_1 = time.time()

        mapper.optimize_map(10, cfg['mapping']['lr_first_factor'], 0, arg_dict["cur_gt_color"], arg_dict["cur_gt_depth"],
                            arg_dict["gt_cur_c2w"], arg_dict["keyframe_dict"], arg_dict["keyframe_list"],
                            arg_dict["cur_c2w"])
        time_ckp_2 = time.time()
        print(f"processing {i + 1} / {n_steps}")
        total_inf_time += (time_ckp_2 - time_ckp_1)
        total_running_time += (time_ckp_2 - time_ckp_0)
        print(f"T_robot : {(time_ckp_2 - time_ckp_1)*1000:.2f} ms, average :{total_inf_time/(i+1)*1000:.2f} ms (GPU computation time on robot)")
        print(f"Service time : {(time_ckp_2 - time_ckp_0)*1000:.2f} ms, average :{total_running_time/(i+1)*1000:.2f} ms")


if __name__ == '__main__':
    main()
