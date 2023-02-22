FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 
#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel 
RUN apt-get update
RUN apt-get install -y vim git
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install einops
RUN pip3 install numba
RUN pip3 install tensorboard
RUN pip3 install torchvision
RUN pip3 install torchaudio
RUN pip3 install pyglet==1.5
RUN pip3 install pandas
RUN pip3 install mat4py
RUN pip3 install scipy
RUN pip3 install torch_scatter
#RUN python3 -m pip install PyQt5 overboard

WORKDIR /work
