FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3 
RUN apt-get update 
RUN apt-get install -y vim git 
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y wget unzip python3-virtualenv 
RUN python3 -m pip install --upgrade pip 

RUN pip3 install matplotlib numpy 
RUN pip3 install pytorch_lightning lpips

WORKDIR /work 

