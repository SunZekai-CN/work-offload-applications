import numpy as np
import time
import pickle
import zmq
import io

port = 5556
map_path = "./src/data_input/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/"
map_files = ["all_u_mat","all_v_mat","all_ui_mat","all_vi_mat","all_Yi","all_s_mat","obstacle_mask"]
output_path = "./src/data_solverOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
output_files= ['policy','value_function']

# create zmq server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")

print('listening...')

times = dict()
map_info_compressed = pickle.loads(socket.recv())
total_begin = time.time()

map_info = []
begin = time.time()
for i in range(len(map_info_compressed)):
    decompressed_array = map_info_compressed[i]
    decompressed_array.seek(0)
    map_info.append(np.load(decompressed_array)["arr_0"])
end = time.time()
times["decompression"] = end - begin
for i in range(len(map_info)):
    np.save(map_path+map_files[i],map_info[i])

# tbegin = time.time()
# result = rt_vibe(frame)
# tend = time.time()
# times['compute'] = tend - tbegin
# print(f'compute {tend-tbegin:.3f}')

output = []
for i in range(len(output_files)):
    output.append(np.load(output_path+output_files[i]+".npy"))

times = dict()

output_compressed = []
begin = time.time()
for i in range(len(output)):
    compressed_array = io.BytesIO()
    np.savez_compressed(compressed_array,output[i])
    output_compressed.append(compressed_array)
end = time.time()
times["compression"] = end - begin

total_end = time.time()
times["on server"] = total_end - total_begin
socket.send(pickle.dumps(output_compressed))

times["on GPU"] = 0
socket.send(pickle.dumps(times))