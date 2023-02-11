
export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_CONFIG=$GVIRTUS_HOME/etc/properties.json
export LD_LIBRARY_PATH=$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH
export LD_PRELOAD="/opt/GVirtuS/lib/frontend/libcudart.so /opt/GVirtuS/lib/frontend/libcublas.so /opt/GVirtuS/lib/frontend/libcudnn.so /opt/GVirtuS/lib/frontend/libcufft.so /opt/GVirtuS/lib/frontend/libcurand.so"
python GPUoffload_test/inference.py --bbox -times 10