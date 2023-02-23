export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=0
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json
export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="/opt/GVirtuS/lib/frontend/libcudart.so /opt/GVirtuS/lib/frontend/libcudnn.so /opt/GVirtuS/lib/frontend/libcufft.so /opt/GVirtuS/lib/frontend/libcurand.so"
# python3 -W ignore test.py
python3 -W ignore GPUoffload_test/inference.py data/gridworld/MazeMap_15x15_vr_2_4000_15_500/models/CALVINConv2d_k_60_i_3_h_150_adam_0.005_0.1_0.25_0214_023552_203490/epoch_000/checkpoint.pt -times 1000 -d cuda:0