import os
import time
import json
import codecs

def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData

times_on_robot=dict()
total_start = time.time()
process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/build_model")
process.read()
process = os.popen("cd src && python3 convert_model_datatype.py && cp temp_modelOp_dirName.txt ../Post_processing/temp_modelOp_dirName.txt")
process.read()
process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/spvi_2")
process.read()
total_end = time.time()
times_on_robot["total time on robot"]=total_end-total_start

f=open(os.path.join("./src/temp_runTime.txt"),"r")
line = f.readline()    
on_gpu=0
while line:              
    lineData=preprocessing(line) 
    if len(lineData)==4 and lineData[2]=="gpu:":
        times_on_robot[lineData[0]]=float(lineData[3])
    line = f.readline() 
f.close()
times_on_robot["build on robot GPU"] = times_on_robot["build"]
times_on_robot["solve on robot GPU"] = times_on_robot["load_model"]+times_on_robot["solve"]


with codecs.open("temp_on_server.json", "r", "utf-8") as f:
    for line in f:
        times_on_server = json.loads(line)

times=dict()
times["total time on robot"]=times_on_robot["total time on robot"]
times["GPU time on robot"]=times_on_robot["build on robot GPU"]+times_on_robot["solve on robot GPU"]
times["GPU time on server"]=times_on_server["build on server GPU"]+times_on_server["solve on server GPU"]
times["offload overhead"] = times_on_server["build offload_overhead"]+times_on_server["solve offload_overhead"]
times["total ad hoc offload"] = times["total time on robot"]-times["GPU time on robot"]+times["GPU time on server"]+times["offload overhead"]
times["speed up ratio"] = times["total ad hoc offload"]/times["total time on robot"]*100
for k,v in times_on_server.items():
    print(f'{k} : {v:.3f}')
for k,v in times_on_robot.items():
    print(f'{k} : {v:.3f}')
for k,v in times.items():
    print(f'{k} : {v:.3f}')