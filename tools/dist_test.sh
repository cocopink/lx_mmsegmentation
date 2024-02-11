CONFIG='/home/lmf/mmsegmentation/work_configs/tamper/tamper_convx_b_exp.py'
CHECKPOINT='/home/lmf/mmsegmentation/work_dirs/tamper/convx_b_exp_6x_lova1_aug1.1_dec1_cpmv/epoch_72.pth'
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --show \
    --show-dir '/home/lmf/mmsegmentation/work_dirs/tamper_convx_b_exp' \
    ${@:4}
#python tools/test.py /home/lmf/mmsegmentation/work_configs/tamper/tamper_convx_b_exp.py /home/lmf/mmsegmentation/work_dirs/tamper/convx_b_exp_6x_lova1_aug1.1_dec1_cpmv/epoch_72.pth --show-dir /home/lmf/mmsegmentation/work_dirs/tamper_convx_b_exp
