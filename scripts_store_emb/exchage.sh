#!/bin/bash

COMMON_ARGS="--root_path ./dataset/exchange/ --data_path exchange.csv --target OT"

python generate_emb_gpt2_custom.py $COMMON_ARGS --flag val
python generate_emb_gpt2_custom.py $COMMON_ARGS --flag train
python generate_emb_gpt2_custom.py $COMMON_ARGS --flag test