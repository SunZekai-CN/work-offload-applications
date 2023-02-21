CURRENT_DIR=$(cd $(dirname $0); pwd)
python ${CURRENT_DIR}/inference.py --bbox -times 1000
