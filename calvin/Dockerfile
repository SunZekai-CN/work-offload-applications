#基于NVIDIA L4T PyTorch基础镜像创建虚拟环境
#官网：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

#初始化镜像
RUN apt-get update
RUN apt-get install -y vim git
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip


#安装calvin所需要的python库
RUN pip3 install matplotlib numpy einops numba tensorboard torchvision torchaudio pyglet==1.5 pandas mat4py scipy torch_scatter

RUN apt-get install -y build-essential libxmu-dev libxi-dev libgl-dev libosmesa-dev liblog4cplus-dev curl
WORKDIR /home
RUN git clone https://github.com/SunZekai-CN/transparent_offload_cuda.git
RUN cd transparent_offload_cuda && mkdir build && cd build && export GVIRTUS_HOME=/opt/GVirtuS && cmake .. && make && make install
#设置初始文件夹
WORKDIR /work