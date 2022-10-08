FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y vim git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv freeglut3-dev wget unzip
RUN python3 -m pip install --upgrade pip

WORKDIR /work

# install requirements
RUN apt-get install -y python3-scipy python3-numba python3-skimage python3-h5py
RUN python3 -m pip install sk-video
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt && rm requirements.txt



COPY . vibe-dx
WORKDIR /work/vibe-dx

