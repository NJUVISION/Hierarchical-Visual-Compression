#!/usr/bin/env bash

set -x
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# set dist args
nproc_per_node=1


nnodes=1
node_rank=0
master_addr=127.0.0.1
master_port=29514

echo "[Using GPUs: $CUDA_VISIBLE_DEVICES, count: ${nproc_per_node}]"

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# set up envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

BED=checkpoints
LOCAL_OUT=local_output
mkdir -p $BED
mkdir -p $LOCAL_OUT

export COMPILE_GAN=0
export USE_TIMELINE_SDK=1
export CUDA_TIMER_STREAM_KAFKA_CLUSTER=bmq_data_va
export CUDA_TIMER_STREAM_KAFKA_TOPIC=megatron_cuda_timer_tracing_original_v2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

swanlab online
exp_name=stage2
bed_path=checkpoints/${exp_name}/
local_out_path=$LOCAL_OUT/${exp_name}

proc_data_path=${PROC_DATA_PATH:-"./data/embedding_mmap"}
proc_res_list='1024'

vae_ckpt=${VAE_CKPT:-"./checkpoints/stage1_final.ckpt"}

rush_resume=${INFINITY_INIT_CKPT:-"./checkpoints/infinity_pretrain.pth"}
rm -rf ${bed_path}

torchrun \
--nproc_per_node=${nproc_per_node} \
--nnodes=${nnodes} \
--node_rank=${node_rank} \
--master_addr=${master_addr} \
--master_port=${master_port} \
train_stage2_var_entropy_flip_prefix.py \
--ep=100 \
--opt=adamw \
--cum=3 \
--sche=lin0 \
--fp16=2 \
--ada=0.9_0.97 \
--tini=-1 \
--tclip=5 \
--flash=0 \
--alng=5e-06 \
--saln=1 \
--cos=1 \
--enable_checkpointing=full-block \
--local_out_path ${local_out_path} \
--task_type='t2i' \
--bed=${bed_path} \
--exp_name=${exp_name} \
--tblr=6e-5 \
--pn 1M \
--model=2bc8 \
--lbs=8 \
--workers=1 \
--Ct5=2048 \
--vae_type 32 \
--vae_ckpt="$vae_ckpt" \
--rush_resume="$rush_resume" \
--wp 0.00000001 \
--wpe=1 \
--dynamic_resolution_across_gpus 1 \
--reweight_loss_by_scale 1 \
--add_lvl_embeding_only_first_block 1 \
--rope2d_each_sa_layer 1 \
--rope2d_normalized_by_hw 2 \
--use_fsdp_model_ema 0 \
--always_training_scales 100 \
--use_bit_label 1 \
--zero=2 \
--save_model_iters_freq 1000 \
--log_freq=50 \
--checkpoint_type='torch' \
--prefetch_factor=16 \
--noise_apply_strength 0.3 \
--noise_apply_layers 13 \
--flip_mode single_boundary \
--apply_spatial_patchify 0 \
--use_flex_attn=True \
--pad=128 \
\
--online_t5=0 \
--use_streaming_dataset=0 \
--dataset_backend='proc_memmap' \
--proc_data_path="$proc_data_path" \
--proc_res_list=${proc_res_list} \
--proc_memmap_cache_size=2 \
\
--enable_student_entropy=0 \
--student_start_step=50000 \
--student_hidden_dim=256 \
--student_depth=4 \
--student_dropout=0.0 \
--student_lr=2e-4 \
--student_wd=1e-4 \
--student_grad_clip=1.0 \
--student_start_scale=1 \
--student_kd_ratio=1.0 \
--student_gt_ratio=1.0 \
--noise_apply_requant 1 \
--random_prefix_min 0 \
--random_prefix_max 12 \
--boundary_flip_prob 1