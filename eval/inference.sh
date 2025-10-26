NNODES=1
NPROC_PER_NODE=8


torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "pretrained_model_path" \
    --revise \
    --output CapMAS/predictions/rediff_s32.json \
    --image_dir CapMAS/images_capmas \
    --steps 32 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "pretrained_model_path" \
    --revise \
    --output CapMAS/predictions/rediff_s128.json \
    --image_dir CapMAS/images_capmas \
    --steps 128 \
