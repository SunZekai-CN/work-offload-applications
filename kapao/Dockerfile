FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt-get update
RUN apt-get install -y vim git
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

#安装Kapao所需要的python库
RUN pip3 install matplotlib numpy Pillow PyYAML scipy tqdm gdown tensorboard seaborn pandas
RUN pip3 install Cython pycocotools thop pytube imageio

RUN apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev liblog4cplus-dev curl liblz4-dev libiberty-dev
WORKDIR /home
RUN git clone https://github.com/SunZekai-CN/transparent_offload_cuda.git
RUN cd transparent_offload_cuda && mkdir build && cd build && export GVIRTUS_HOME=/opt/GVirtuS && cmake .. && make && make install

WORKDIR /work

# Let's not copy the directory, as the data is huge
# Mount it instead
# COPY nice-slam .
