NNODES=1
NPROC_PER_NODE=8
# NPROC_PER_NODE=4

# 启动命令

# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=127.0.0.1 \
#     --master_port=29500 \
#     --node_rank=0 \
#     inference.py \
#     --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_15/checkpoint-4000" \
#     --output /group/40005/auroraji/CAPTURE/predictions/llada_v_finetune_15_s16.json \
#     --image_dir /group/40005/auroraji/CAPTURE/samples \
#     --steps 16 \

# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=127.0.0.1 \
#     --master_port=29500 \
#     --node_rank=0 \
#     inference.py \
#     --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_15/checkpoint-4000" \
#     --output /group/40005/auroraji/CAPTURE/predictions/llada_v_finetune_15_s32.json \
#     --image_dir /group/40005/auroraji/CAPTURE/samples \
#     --steps 32 \

# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=127.0.0.1 \
#     --master_port=29500 \
#     --node_rank=0 \
#     inference.py \
#     --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_15/checkpoint-4000" \
#     --output /group/40005/auroraji/CAPTURE/predictions/llada_v_finetune_15_s64.json \
#     --image_dir /group/40005/auroraji/CAPTURE/samples \
#     --steps 64 \

# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=127.0.0.1 \
#     --master_port=29500 \
#     --node_rank=0 \
#     inference.py \
#     --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_15/checkpoint-4000" \
#     --output /group/40005/auroraji/CAPTURE/predictions/llada_v_finetune_15_s128.json \
#     --image_dir /group/40005/auroraji/CAPTURE/samples \
#     --steps 128 \

# torchrun \
#     --nnodes=$NNODES \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=127.0.0.1 \
#     --master_port=29500 \
#     --node_rank=0 \
#     inference.py \
#     --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_19" \
#     --output /group/40005/auroraji/CAPTURE/predictions/llada_v_finetune_19_s128.json \
#     --image_dir /group/40005/auroraji/CAPTURE/samples \
#     --steps 128 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapArena/predictions/llada_v_finetune_22_s16.json \
    --steps 16 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapArena/predictions/llada_v_finetune_22_s32.json \
    --steps 32 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapArena/predictions/llada_v_finetune_22_s64.json \
    --steps 64 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapArena/predictions/llada_v_finetune_22_s128.json \
    --steps 128 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapMAS/predictions/llada_v_finetune_22_s32.json \
    --image_dir /group/40005/auroraji/CapMAS/images_capmas \
    --steps 32 \

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    --node_rank=0 \
    inference.py \
    --pretrained "/group/40005/auroraji/LLaDA-V/train/exp/llada_v_finetune_22" \
    --output /group/40005/auroraji/CapMAS/predictions/llada_v_finetune_22_s128.json \
    --image_dir /group/40005/auroraji/CapMAS/images_capmas \
    --steps 128 \


python /group/40005/auroraji/burn.py