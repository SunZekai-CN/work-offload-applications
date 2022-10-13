import os
import time
import json
import codecs

def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData

total_start = time.time()
process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/build_model")
# print(process.read())
process = os.popen("cd src && python3 convert_model_datatype.py && cp temp_modelOp_dirName.txt ../Post_processing/temp_modelOp_dirName.txt")
# print(process.read())
process = os.popen("cd src && CUDA_VISIBLE_DEVICES=0 ./bin/spvi_2")
# print(process.read())
total_end = time.time()
total_time=total_end-total_start

f=open(os.path.join("./src/temp_runTime.txt"),"r")
line = f.readline()    
on_gpu=0
while line:              
    lineData=preprocessing(line) 
    if len(lineData)==4 and lineData[2]=="gpu:":
        on_gpu+=float(lineData[3])
    line = f.readline() 
f.close()
print("total time on robot:", total_time)
print("GPU time on robot:",on_gpu)



with codecs.open("temp_on_server.json", "r", "utf-8") as f:
    for line in f:
        times_on_server = json.loads(line)
print("GPU time on server:",times_on_server["on server GPU"])
for k in ["offload_overhead","send+receive","compression on client","decompression on server","compression on server","decompression on client"]:
    print(f'{k} : {times_on_server[k]:.3f}')
ad_hoc_offload = total_time - on_gpu + times_on_server["on server GPU"] + times_on_server["offload_overhead"]
print(f"total ad hoc offload: {ad_hoc_offload} , {ad_hoc_offload/total_time*100:.2f}%")