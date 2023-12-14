#!/usr/bin/env bash

RUN_NAME="imitation_learning"

MIN_SIZE=20  # job number
MACHINE=5    # mahine number 

N_EPOCHS=55  # training epochs 
EPOCH_SIZE=12800  # training samples 
BATCH_SIZE=128
VAL_SIZE=1280  # testing samples 
ROLLOUT_SIZE=1280

MODEL="attention"
ENCODER="gnn"
AGGREGATION="max"
AGGREGATION_GRAPH="mean"
NORMALIZATION="batch"
EMBEDDING_DIM=128
N_ENCODE_LAYERS=3
ACCUMULATION_STEPS=1

PROBLEM="tspsl"
DEVICES="0"
NUM_WORKERS=0
NEIGHBORS=0.2
KNN_STRAT="percentage"

LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0

CUDA_VISIBLE_DEVICES="$DEVICES" python run.py --problem "$PROBLEM" \
    --model "$MODEL" \
    --min_size "$MIN_SIZE" --machine "$MACHINE" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --train_dataset "$TRAIN_DATASET" \
    --val_datasets "$VAL_DATASET1" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --val_size "$VAL_SIZE" --rollout_size "$ROLLOUT_SIZE" \
    --encoder "$ENCODER" --aggregation "$AGGREGATION" \
    --n_encode_layers "$N_ENCODE_LAYERS" --gated \
    --normalization "$NORMALIZATION" --learn_norm \
    --embedding_dim "$EMBEDDING_DIM" --hidden_dim "$EMBEDDING_DIM" \
    --lr_model "$LR_MODEL" --max_grad_norm "$MAX_NORM" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
    --run_name "$RUN_NAME"