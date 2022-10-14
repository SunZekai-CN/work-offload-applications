import numpy as np
import time
import pickle
import zmq
import io
import os

port = 5556
map_path = "./src/data_input/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/"
map_files = ["all_u_mat","all_v_mat","all_ui_mat","all_vi_mat","all_Yi","all_s_mat","obstacle_mask"]
model_path = "./src/data_modelOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
model_files = ["DP_relv_params","master_cooS2","master_R","master_cooS1","master_cooVal","prob_params"]
output_path = "./src/data_solverOutput/custom1/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/a2x16_i15_j40_ref2/0.050000/"
output_files= ['policy','value_function']

def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData

# create zmq server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")

print('listening...')

times = dict()
map_info_compressed = pickle.loads(socket.recv())
total_begin = time.time()
print("i get and building model...")

map_info = []
begin = time.time()
map_info_compressed.seek(0)
map_info_compressed = np.load(map_info_compressed)
for i in range(len(map_files)):
    map_info.append(map_info_compressed[map_files[i]])
end = time.time()
times["build decompression"] = end - begin
for i in range(len(map_files)):
    np.save(map_path+map_files[i],map_info[i])


process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/build_model")
process.read()
print("i finish building models...")
output = []
for i in range(len(model_files)):
    output.append(np.load(model_path+model_files[i]+".npy"))
output_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(output_compressed,DP_relv_params=output[0],master_cooS2=output[1],master_R=output[2],master_cooS1=output[3],master_cooVal=output[4],prob_params=output[5])
end = time.time()
times["build compression"] = end - begin

total_end = time.time()
times["build on server"] = total_end - total_begin

socket.send(pickle.dumps(output_compressed))
models_compressed = pickle.loads(socket.recv())
total_begin = time.time()
print("i get and solving...")

models = []
begin = time.time()
models_compressed.seek(0)
models_compressed = np.load(models_compressed)
for i in range(len(model_files)):
    models.append(models_compressed[model_files[i]])
end = time.time()
times["solve decompression"] = end - begin
for i in range(len(model_files)):
    np.save(model_path+model_files[i],models[i])

process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/spvi_2")
process.read()
print("i finish solving...")
output = []
for i in range(len(output_files)):
    output.append(np.load(output_path+output_files[i]+".npy"))


output_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(output_compressed,policy=output[0],value_function=output[1])
end = time.time()
times["solve compression"] = end - begin

total_end = time.time()
times["solve on server"] = total_end - total_begin
socket.send(pickle.dumps(output_compressed))

f=open(os.path.join("./src/temp_runTime.txt"),"r")
line = f.readline()    
while line:              
    lineData=preprocessing(line) 
    if len(lineData)==4 and lineData[2]=="gpu:":
        times[lineData[0]]=float(lineData[3])
    line = f.readline() 
f.close()
times["solve on server GPU"]=times["solve"]+times["load_model"] 
times["build on server GPU"]=times["build"]

pickle.loads(socket.recv())
socket.send(pickle.dumps(times))