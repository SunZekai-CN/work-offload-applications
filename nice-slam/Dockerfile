FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel 
RUN apt-get update 
RUN apt-get install -y vim git 
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv 
RUN python3 -m pip install --upgrade pip 

RUN pip3 install matplotlib numpy==1.21.2 
RUN pip3 install colorama open3d scikit-image trimesh
RUN pip3 install pyyaml tqdm

WORKDIR /work 
