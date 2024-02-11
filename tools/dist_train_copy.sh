CONFIG='/home/lmf/mmsegmentation/work_configs/tamper/tamper_convx_l_copy.py'
GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29508}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CUDA_VISIBLE_DEVICES="2,3" \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
