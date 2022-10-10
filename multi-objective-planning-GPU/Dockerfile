FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3

RUN apt-get update
RUN apt-get install -y vim git
#RUN apt-get install libopenexr-dev
RUN DEBIAN_FRONTEND=oninteractive apt-get install -y python3-opencv wget unzip
RUN python3 -m pip install --upgrade pip
  
RUN pip install addict anyio argon2-cffi argon2-cffi-bindings argparse attrs babel backcall beautifulsoup4 bleach charset-normalizer cloudpickle \
    && pip install colorama dask debugpy decorator defusedxml deprecation entrypoints filelock fsspec gdown imageio imath importlib-metadata \
    && pip install importlib-resources jedi jinja2 joblib json5 jsonschema \
    && pip install numpy open3d pandas scikit-image scikit-learn scipy seaborn tqdm urllib3

RUN apt-get install -y python3-rtree
RUN python3 -m pip install pyzmq
RUN python3 -m pip install trimesh mathutils==2.81.2

WORKDIR /work

# Let's not copy the directory, as the data is huge
# Mount it instead
# COPY nice-slam .
