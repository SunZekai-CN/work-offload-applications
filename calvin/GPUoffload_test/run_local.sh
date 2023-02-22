CURRENT_DIR=$(cd $(dirname $0); pwd)
python3 ${CURRENT_DIR}/inference.py data/gridworld/MazeMap_15x15_vr_2_4000_15_500/models/CALVINConv2d_k_60_i_3_h_150_adam_0.005_0.1_0.25_0214_023552_203490/epoch_000/checkpoint.pt -times 1000 -d cuda:0
