# FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
RUN apt-get update
RUN apt-get install -y vim git
#RUN apt-get install libopenexr-dev
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y python3-opencv wget unzip python3-virtualenv
RUN python3 -m pip install --upgrade pip

RUN pip install matplotlib numpy opencv-python Pillow PyYAML scipy tqdm gdown tensorboard seaborn pandas
RUN pip install Cython pycocotools thop


WORKDIR /work

# Let's not copy the directory, as the data is huge
# Mount it instead
# COPY nice-slam .
