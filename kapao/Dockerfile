# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
RUN apt-get update
RUN apt-get install -y vim git
#RUN apt-get install libopenexr-dev
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y python3-opencv wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

RUN apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev liblog4cplus-dev curl
WORKDIR /home
RUN git clone https://github.com/SunZekai-CN/transparent_offload_cuda.git
RUN cd transparent_offload_cuda && mkdir build && cd build && export GVIRTUS_HOME=/opt/GVirtuS && cmake .. && make && make install


RUN pip install matplotlib numpy opencv-python Pillow PyYAML scipy tqdm gdown tensorboard seaborn pandas
RUN pip install Cython pycocotools thop pytube imageio


WORKDIR /work

# Let's not copy the directory, as the data is huge
# Mount it instead
# COPY nice-slam .
