NNODES=1
NPROC_PER_NODE=8  # 使用8张GPU

# 启动命令
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "pretrained_model_path" \
    --steps 32 \
    --output rediff_base_s32_outputs_vicrit_others.json \
    --revise

