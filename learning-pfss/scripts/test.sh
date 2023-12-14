#!/usr/bin/env bash

RUN_NAME="test" 

MODEL="outputs/pfss_20/imitation_learning_20220528T155149/epoch-54.pt"

MIN_SIZE=20  # job number
MACHINE=5    # mahine number 
VAL_SIZE=1000 # testing samples 
BATCH_SIZE=100

PROBLEM="tspsl"
DEVICES="0,1"
NUM_WORKERS=0
NEIGHBORS=0.2
KNN_STRAT="percentage"
N_EPOCHS=100
EPOCH_SIZE=12800
ACCUMULATION_STEPS=1
ENCODER="gnn"
ROLLOUT_SIZE=1000
AGGREGATION="max"
AGGREGATION_GRAPH="mean"
NORMALIZATION="batch"
EMBEDDING_DIM=128
N_ENCODE_LAYERS=3
LR_MODEL=0.0001
MAX_NORM=1
CHECKPOINT_EPOCHS=0

python run.py --problem "$PROBLEM" \
    --model "$MODEL" --baseline "$BASELINE" \
    --min_size "$MIN_SIZE"  --machine "$MACHINE" \
    --neighbors "$NEIGHBORS" --knn_strat "$KNN_STRAT" \
    --val_datasets "$VAL_DATASET1" \
    --epoch_size "$EPOCH_SIZE" \
    --batch_size "$BATCH_SIZE" --accumulation_steps "$ACCUMULATION_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --val_size "$VAL_SIZE" --rollout_size "$ROLLOUT_SIZE" \
    --encoder "$ENCODER" --aggregation "$AGGREGATION" --aggregation_graph "$AGGREGATION_GRAPH" \
    --n_encode_layers "$N_ENCODE_LAYERS" --gated \
    --normalization "$NORMALIZATION" --learn_norm \
    --embedding_dim "$EMBEDDING_DIM" --hidden_dim "$EMBEDDING_DIM" \
    --lr_model "$LR_MODEL" --max_grad_norm "$MAX_NORM" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_epochs "$CHECKPOINT_EPOCHS" \
    --run_name "$RUN_NAME" --eval_only