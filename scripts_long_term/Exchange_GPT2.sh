# ------------- 基础设置 -------------dos2unix /home/yuanyuan/BALM-TSF/scripts_long_term/BALM_Exchange_GPT2.sh
export CUDA_VISIBLE_DEVICES=0        # 可改为 0,1,2...
export TOKENIZERS_PARALLELISM=false  # 禁用多进程 tokenizer 警告
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=YOUR_API_KEY
export WANDB_MODE=offline   # 可选：仅本地记录，不同步云端
model_name=BALM
train_epochs=3
learning_rate=0.005
llama_layers=6 # 6 for GPT2-small, 32 for llama
llm_dim=768
llm_model=GPT2

master_port=29594
num_process=1 # 1 for single GPU, 2 for two GPUs
batch_size=24
d_model=32
d_ff=128

comment='BALM-Exchange'
wd_project=BALM-Exchange
version_num=BALM-Exchange
target="OT"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange/ \
  --data_path exchange.csv \
  --model_id exchange_512_96 \
  --model $model_name \
  --data exchange\
  --features M \
  --target "OT" \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --llm_model $llm_model\
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange/ \
  --data_path exchange.csv \
  --model_id exchange_512_192 \
  --model $model_name \
  --data exchange \
  --features M \
  --target "OT" \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --llm_model $llm_model\
  --llm_dim $llm_dim \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange/ \
  --data_path exchange.csv \
  --model_id exchange_512_336 \
  --model $model_name \
  --data exchange \
  --features M \
  --target "OT" \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --llm_model $llm_model\
  --llm_dim $llm_dim \
  --train_epochs 3 \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num

  accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/exchange/ \
  --data_path exchange.csv \
  --model_id exchange_512_720 \
  --model $model_name \
  --data exchange \
  --features M \
  --target "OT" \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --llm_model $llm_model\
  --llm_dim $llm_dim \
  --train_epochs 3 \
  --model_comment $comment \
  --wandb_flag 1 \
  --wd_project $wd_project \
  --version_num $version_num