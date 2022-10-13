import numpy as np
import time
import pickle
import zmq
import io
import os

port = 5556
map_path = "./src/data_input/DG3_g100x100x120_r5k_vmax6_2LpDynObs_0.4_0.1_exp_3s_2r/"
map_files = ["all_u_mat","all_v_mat","all_ui_mat","all_vi_mat","all_Yi","all_s_mat","obstacle_mask"]
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

map_info = []
begin = time.time()
map_info_compressed.seek(0)
map_info_compressed = np.load(map_info_compressed)
for i in range(len(map_files)):
    map_info.append(map_info_compressed[map_files[i]])
end = time.time()
times["decompression"] = end - begin
for i in range(len(map_info)):
    np.save(map_path+map_files[i],map_info[i])


process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/build_model")
# print(process.read())
process = os.popen("cd src && python3 convert_model_datatype.py && cp temp_modelOp_dirName.txt ../Post_processing/temp_modelOp_dirName.txt")
# print(process.read())
process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/spvi_2")
# print(process.read())

output = []
for i in range(len(output_files)):
    output.append(np.load(output_path+output_files[i]+".npy"))


output_compressed = io.BytesIO()
begin = time.time()
np.savez_compressed(output_compressed,policy=output[0],value_function=output[1])
end = time.time()
times["compression"] = end - begin

total_end = time.time()
times["on server"] = total_end - total_begin
socket.send(pickle.dumps(output_compressed))

f=open(os.path.join("./src/temp_runTime.txt"),"r")
line = f.readline()    
on_gpu=0
while line:              
    lineData=preprocessing(line) 
    if len(lineData)==4 and lineData[2]=="gpu:":
        on_gpu+=float(lineData[3])
        print(lineData)
    line = f.readline() 
f.close()

times["on GPU"] = on_gpu
socket.send(pickle.dumps(times))