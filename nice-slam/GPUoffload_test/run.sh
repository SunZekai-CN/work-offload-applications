CURRENT_DIR=$(cd $(dirname $0); pwd)
python -W ignore ${CURRENT_DIR}/inference.py configs/Demo/demo.yaml --times 100