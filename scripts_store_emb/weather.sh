#!/bin/bash

COMMON_ARGS="--root_path ./dataset/weather/ --data_path weather.csv --target OT"

python generate_emb_gpt2.py $COMMON_ARGS --flag val
python generate_emb_gpt2.py $COMMON_ARGS --flag train
python generate_emb_gpt2.py $COMMON_ARGS --flag test