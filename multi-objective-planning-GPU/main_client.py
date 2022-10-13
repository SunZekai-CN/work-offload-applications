import time
import pickle
import numcodecs
import zmq
import numpy as np
import sys
import io
import json
import codecs

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


map_info_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(map_info_compressed,all_u_mat=map_info[0],all_v_mat=map_info[1],all_ui_mat=map_info[2],all_vi_mat=map_info[3],all_Yi=map_info[4],all_s_mat=map_info[5],obstacle_mask=map_info[6])
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
output.seek(0)
output = np.load(output)
for i in range(len(output_files)):
    output_decompressed.append(output[output_files[i]])
end = time.time()
times["decompression on client"] = end - begin
for i in range(len(output_files)):
    np.save(output_path +output_files[i],output_decompressed[i])

# times_on_server = pickle.loads(socket.recv())
times_on_server = dict()
times_on_server["on server"] = 1.0
times_on_server["on GPU"] = 0.5
times_on_server["decompression"] = 0.1
times_on_server["compression"] = 0.2

times["on server"] = times_on_server["on server"]
times["send+receive"] = times["request"]-times["on server"]
times["on server GPU"] = times_on_server["on GPU"]
times["decompression on server"] = times_on_server["decompression"]
times["compression on server"] = times_on_server["compression"]
times["offload_overhead"] = times["compression on client"] + times["decompression on server"] + times["compression on server"]+times["decompression on client"]+times["send+receive"]


for k in ["on server GPU","offload_overhead","send+receive","compression on client","decompression on server","compression on server","decompression on client"]:
    print(f'{k} : {times[k]:.3f}')


with codecs.open('temp_on_server.json','a', 'utf-8') as outf:
    json.dump(times, outf, ensure_ascii=False)
