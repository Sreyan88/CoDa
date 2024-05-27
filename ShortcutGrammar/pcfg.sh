#!/bin/bash -l

if [ -z "$1" ]; then
    echo "Error: DATASET not provided. Please provide the DATASET as an argument."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: NUM_GPUS not provided. Please provide the NUM_GPUS as an argument."
    exit 1
fi

DATASET=$1
PORT_ID=$(expr $RANDOM + 1000)
NUM_GPU=$2
export OMP_NUM_THREADS=$2
export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load cuda

python -m torch.distributed.launch --use_env --nproc_per_node $NUM_GPU --master_port $PORT_ID src/run.py \
    --train_on "${DATASET}" \
    --model_type pcfg \
    --max_length 128 \
    --max_eval_length 128 \
    --min_length 1 \
    --use_product_length 0 \
    --epochs 40 \
    --patience 5 \
    --eval_before_training \
    --eval_every 4096 \
    --eval_before_training \
    --eval_on "${DATASET}" \
    --eval_on_train_datasets "${DATASET}" \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --criterion loss \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --save \
    --preterminals 64 \
    --nonterminals 32 \
    --output_dir "output/${DATASET}/pcfg";
