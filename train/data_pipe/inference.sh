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
    --pretrained "/group/40005/auroraji/pretrained/LLaDA-V" \
    --steps 32 \
    --output llada_v_s32_outputs_vicrit_others.json \
    --revise "False"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/pretrained/LLaDA-V" \
    --steps 128 \
    --output llada_v_s128_outputs_vicrit_others.json \
    --revise "False"

python /group/40005/auroraji/burn.py