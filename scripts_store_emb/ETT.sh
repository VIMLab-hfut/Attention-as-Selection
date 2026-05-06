#!/bin/bash

COMMON_ARGS="--root_path ./dataset/ETT-small/ --data_path ETTh1.csv --target OT"

python generate_emb_gpt2_ETT.py $COMMON_ARGS --flag val
python generate_emb_gpt2_custom_ETT.py $COMMON_ARGS --flag train
python generate_emb_gpt2_custom_ETT.py $COMMON_ARGS --flag test
python generate_emb_gpt2_custom_ETT.py $COMMON_ARGS --flag test
