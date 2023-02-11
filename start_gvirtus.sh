export GVIRTUS_HOME=/opt/GVirtuS
export GVIRTUS_LOGLEVEL=0
vim $GVIRTUS_HOME/etc/properties.json
LD_LIBRARY_PATH=$GVIRTUS_HOME/lib:$LD_LIBRARY_PATH $GVIRTUS_HOME/bin/gvirtus-backend $GVIRTUS_HOME/etc/properties.json