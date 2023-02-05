# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y vim git
#RUN apt-get install libopenexr-dev
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y python3-opencv wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

RUN apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev liblog4cplus-dev curl cmake
WORKDIR /home
RUN git clone https://github.com/SunZekai-CN/transparent_offload_cuda.git
RUN cd transparent_offload_cuda && mkdir build && cd build && export GVIRTUS_HOME=/opt/GVirtuS && cmake .. && make && make install


WORKDIR /work

# Let's not copy the directory, as the data is huge
# Mount it instead
# COPY nice-slam .
