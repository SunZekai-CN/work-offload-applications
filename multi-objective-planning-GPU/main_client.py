import time
import pickle
import numcodecs
import zmq
import numpy as np
import sys
import io
from collections import OrderedDict

addr = "172.22.2.7:5556"
map_path = "./src/data_input/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/"
map_files = ["all_u_mat","all_v_mat","all_ui_mat","all_vi_mat","all_Yi","all_s_mat","obstacle_mask"]
output_path = "./src/data_solverOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
output_files= ['policy','value_function']



# connect zmq server
# context = zmq.Context()
# socket = context.socket(zmq.REQ)
# socket.connect(f"tcp://{addr}")
# time.sleep(1)

map_info = []
for i in range(len(map_files)):
    map_info.append(np.load(map_path+map_files[i]+".npy"))

times = dict()

map_info_compressed = []
begin = time.time()
for i in range(len(map_info)):
    compressed_array = io.BytesIO()
    np.savez_compressed(compressed_array,map_info[i])
    map_info_compressed.append(compressed_array)
end = time.time()
times["compression on client"] = end - begin

begin = time.time()
# socket.send(pickle.dumps(map_info_compressed))
# output = pickle.loads(socket.recv())
output = map_info_compressed
end = time.time()
times["request"] = end -begin


output_decompressed = []
begin = time.time()
for i in range(len(output)):
    decompressed_array = output[i]
    decompressed_array.seek(0)
    output_decompressed.append(np.load(decompressed_array)["arr_0"])
end = time.time()
times["decompression on client"] = end - begin
for i in range(len(output_decompressed)):
    np.save(output_path+output_files[i],output_decompressed[i])

# times_on_server = pickle.loads(socket.recv())
times_on_server = dict()
times_on_server["on server"] = 1.0
times_on_server["on GPU"] = 0.5
times_on_server["decompression"] = 0.1
times_on_server["compression"] = 0.2

times["on server"] = times_on_server["on server"]
times["send+receive"] = times["request"]-times["on server"]
times["compute"] = times_on_server["on GPU"]
times["decompression on server"] = times_on_server["decompression"]
times["compression on server"] = times_on_server["compression"]
times["offload_overhead"] = times["compression on client"] + times["decompression on server"] + times["compression on server"]+times["decompression on client"]+times["send+receive"]


for k in ["compute","offload_overhead","send+receive","compression on client","decompression on server","compression on server","decompression on client"]:
    print(f'{k} : {times[k]:.3f}')