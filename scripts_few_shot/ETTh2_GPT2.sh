# ------------- 基础设置 -------------
export CUDA_VISIBLE_DEVICES=0        # 可改为 0,1,2...
export TOKENIZERS_PARALLELISM=false  # 禁用多进程 tokenizer 警告
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=YOUR_API_KEY
export WANDB_MODE=offline   # 可选：仅本地记录，不同步云端
model_name=BALM
train_epochs=30
learning_rate=0.01
llama_layers=6 # 6 for GPT2-small, 32 for llama
llm_dim=768
llm_model=GPT2

master_port=29504
num_process=1 # 1 for single GPU, 2 for two GPUs
batch_size=8
d_model=32
d_ff=128

comment='BALM-ETTh1'
wd_project=BALM-fewshot
version_num=BALM-fewshot

percent=15


  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \

accelerate launch --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 64 \
  --d_ff 128 \
  --batch_size 8 \
  --learning_rate 0.001\
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \

accelerate launch --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 64 \
  --d_ff 64 \
  --batch_size 8 \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \


accelerate launch --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs 20 \
  --patience 10 \
  --model_comment $comment \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num \
  --percent $percent \