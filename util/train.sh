MASTER_ADDR=${MASTER_ADDR:?need master addr from platform}
MASTER_PORT=${MASTER_PORT:?need master port from platform}

NUM_MACHINES=${WORLD_SIZE:?need node count from platform}
NODE_RANK=${RANK:?need node rank from platform}
GPU_PER_NODE=${GPU_NUM_PER_NODE:-8}
NUM_PROCESSES=$((NUM_MACHINES * GPU_PER_NODE))

CACHE_ROOT=/file_system/goosefsx/whu/users/hong.li/.cache

export PYTHONPATH=/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/dinov2-main:$PYTHONPATH
export TORCH_HOME=$CACHE_ROOT/torch
export HF_HOME=$CACHE_ROOT/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1

# 动态模块缓存放本地盘，避免共享盘并发写坏
export HF_MODULES_CACHE=/file_system/goosefsx/whu/users/hong.li/qizhe/checkpoints/tmp/hf_modules_${HOSTNAME}
mkdir -p ${HF_MODULES_CACHE}

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "HOSTNAME=${HOSTNAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "RANK=${RANK}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "NODE_RANK=${NODE_RANK}"
echo "GPU_PER_NODE=${GPU_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "HF_MODULES_CACHE=${HF_MODULES_CACHE}"

source /file_system/goosefsx/whu/users/hong.li/miniconda3/bin/activate
conda activate foundation_stereo
accelerate launch \
  --multi_gpu \
  --num_processes ${NUM_PROCESSES} \
  --num_machines ${NUM_MACHINES} \
  --machine_rank ${NODE_RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --mixed_precision no \
  --dynamo_backend no \
  train_GeoStereo.py \
  --output_dir  root2/ckpt  \
  --logging_dir  root2/log  \
  --vis_step  10  \