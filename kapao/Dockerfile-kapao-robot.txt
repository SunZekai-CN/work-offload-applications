#基于从NVIDIA官网上下载的 L4T PyTorch基础镜像（https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch）创建虚拟环境
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

#初始化镜像
RUN apt-get update
RUN apt-get install -y vim git
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

#安装Kapao所需要的python库
RUN pip3 install matplotlib numpy Pillow PyYAML scipy tqdm gdown tensorboard seaborn pandas
RUN pip3 install Cython pycocotools thop pytube imageio

#设置初始文件夹
WORKDIR /work


