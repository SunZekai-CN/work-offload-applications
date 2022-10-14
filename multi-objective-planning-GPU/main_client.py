import time
import pickle
import zmq
import numpy as np
import sys
import io
import json
import codecs
import os

addr = "172.22.100.186:5556"
# addr = "172.22.2.7:5556"
map_path = "./src/data_input/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/"
map_files = ["all_u_mat","all_v_mat","all_ui_mat","all_vi_mat","all_Yi","all_s_mat","obstacle_mask"]
model_path = "./src/data_modelOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
model_files = ["DP_relv_params","master_cooS2","master_R","master_cooS1","master_cooVal","prob_params"]
output_path = "./src/data_solverOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
output_files= ['policy','value_function']

# connect zmq server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{addr}")
time.sleep(1)

map_info = []
for i in range(len(map_files)):
    map_info.append(np.load(map_path+map_files[i]+".npy"))

times = dict()


map_info_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(map_info_compressed,all_u_mat=map_info[0],all_v_mat=map_info[1],all_ui_mat=map_info[2],all_vi_mat=map_info[3],all_Yi=map_info[4],all_s_mat=map_info[5],obstacle_mask=map_info[6])
end = time.time()
times["build compression on client"] = end - begin

print("i send and waiting...")
begin = time.time()
socket.send(pickle.dumps(map_info_compressed))
models_compressed = pickle.loads(socket.recv())
end = time.time()
times["build request"] = end -begin


models = []
begin = time.time()
models_compressed.seek(0)
models_compressed = np.load(models_compressed)
for i in range(len(model_files)):
    models.append(models_compressed[model_files[i]])
end = time.time()
times["build decompression on client"] = end - begin
for i in range(len(model_files)):
    np.save(model_path+model_files[i],models[i])

process = os.popen("cd src && python3 convert_model_datatype.py && cp temp_modelOp_dirName.txt ../Post_processing/temp_modelOp_dirName.txt")
process.read()

models = []
for i in range(len(model_files)):
    models.append(np.load(model_path+model_files[i]+".npy"))
model_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(model_compressed,DP_relv_params=models[0],master_cooS2=models[1],master_R=models[2],master_cooS1=models[3],master_cooVal=models[4],prob_params=models[5])
end = time.time()
times["solve compression on client"] = end - begin

print("i send and waiting...")
begin = time.time()
socket.send(pickle.dumps(model_compressed))
output = pickle.loads(socket.recv())
end = time.time()
times["solve request"] = end -begin

output_decompressed = []
begin = time.time()
output.seek(0)
output = np.load(output)
for i in range(len(output_files)):
    output_decompressed.append(output[output_files[i]])
end = time.time()
times["solve decompression on client"] = end - begin
for i in range(len(output_files)):
    np.save(output_path +output_files[i],output_decompressed[i])

socket.send(pickle.dumps("ok"))
times_on_server = pickle.loads(socket.recv())

times["build on server"] = times_on_server["build on server"]
times["build send+receive"] = times["build request"]-times["build on server"]
times["build on server GPU"] = times_on_server["build on server GPU"]
times["build decompression on server"] = times_on_server["build decompression"]
times["build compression on server"] = times_on_server["build compression"]
times["build offload_overhead"] = times["build compression on client"] + times["build decompression on server"] + times["build compression on server"]+times["build decompression on client"]+times["build send+receive"]
times["solve on server"] = times_on_server["solve on server"]
times["solve send+receive"] = times["solve request"]-times["solve on server"]
times["solve on server GPU"] = times_on_server["solve on server GPU"]
times["solve decompression on server"] = times_on_server["solve decompression"]
times["solve compression on server"] = times_on_server["solve compression"]
times["solve offload_overhead"] = times["solve compression on client"] + times["solve decompression on server"] + times["solve compression on server"]+times["solve decompression on client"]+times["solve send+receive"]

print(times)

with codecs.open('temp_on_server.json','w', 'utf-8') as outf:
    json.dump(times, outf, ensure_ascii=False)
