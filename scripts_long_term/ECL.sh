#!/bin/bash

# ------------- 基础设置 -------------
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export WANDB_MODE=offline

# ------------- 统一公共参数 -------------
model_name=BALM
train_epochs=50
learning_rate=1e-3
llama_layers=6
llm_dim=768
llm_model=GPT2
target="MT_320"
batch_size=8
d_model=128
d_ff=256

comment='BALM-ECL'
wd_project=BALM-ECL
version_num=BALM-ECL

# dataset 设置 - 修正为单变量
root_path=./dataset/ECL/
data_path=ECL.csv
seq_len=512
label_len=48
features=S          # 单变量
enc_in=7            # 修正：输入特征数为1
dec_in=7            # 修正：解码器输入特征数为1
c_out=1             # 输出特征数为1

# ------------- 预测长度列表 -------------
pred_list=(96 192 336 720)

for pred_len in "${pred_list[@]}"
do
    echo "========================================"
    echo "开始训练：pred_len=${pred_len}"
    echo "========================================"

    python run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ECL_${seq_len}_${pred_len} \
    --model $model_name \
    --data ECL \
    --features $features \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --des 'Exp' \
    --itr 1 \
    --llm_model $llm_model \
    --llm_dim $llm_dim \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment \
    --wandb_flag 0 \
    --wd_project $wd_project \
    --version_num ${version_num}_${pred_len}

    echo "========================================"
    echo "pred_len=${pred_len} 训练完成"
    echo "========================================"
done

echo "========================================"
echo "全部步长训练完成！"
echo "========================================"
