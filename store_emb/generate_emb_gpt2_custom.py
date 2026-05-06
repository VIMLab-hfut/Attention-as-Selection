#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import h5py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import argparse

def time_features(dates, freq='h'):
    features = []
    if freq == 'h':
        features.append(np.array([d.hour for d in dates]))
        features.append(np.array([d.day for d in dates]))
        features.append(np.array([d.month for d in dates]))
        features.append(np.array([d.weekday() for d in dates]))
    elif freq == 'd':
        features.append(np.array([d.day for d in dates]))
        features.append(np.array([d.month for d in dates]))
        features.append(np.array([d.weekday() for d in dates]))
    elif freq == 'm':
        features.append(np.array([d.month for d in dates]))
        features.append(np.array([d.year for d in dates]))
    return np.vstack(features)

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./dataset/weather/")
parser.add_argument("--data_path", type=str, default="weather.csv")
parser.add_argument("--flag", type=str, choices=["train","val","test"], default="test")
parser.add_argument("--save_dir", type=str, default="./Embeddings")
parser.add_argument("--features", type=str, choices=["S","M","MS"], default="M")
parser.add_argument("--target", type=str, default="OT")
parser.add_argument("--scale", type=bool, default=True)
parser.add_argument("--timeenc", type=int, choices=[0,1], default=0)
parser.add_argument("--freq", type=str, default="h")
parser.add_argument("--percent", type=int, default=100)
parser.add_argument("--seq_len", type=int, default=36)
parser.add_argument("--label_len", type=int, default=48)
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--llm_dim", type=int, default=768)   
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


if not args.data_path.endswith('.csv'):
    args.data_path += '.csv'
data_path_file = args.data_path[:-4]  
save_embed_path = os.path.join(args.save_dir, data_path_file, args.flag)
os.makedirs(save_embed_path, exist_ok=True)


config = GPT2Config.from_pretrained("openai-community/gpt2")
gpt2 = GPT2Model.from_pretrained(
    "openai-community/gpt2",
    config=config,
    torch_dtype=torch.float16
).to(args.device)

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2.eval()
for p in gpt2.parameters():
    p.requires_grad = False


scaler = StandardScaler()
df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))


cols = list(df_raw.columns)
cols.remove(args.target)
cols.remove("date")
df_raw = df_raw[["date"] + cols + [args.target]]


num_train = int(len(df_raw) * 0.7)
num_test = int(len(df_raw) * 0.2)
num_vali = len(df_raw) - num_train - num_test
type_map = {"train":0, "val":1, "test":2}
set_type = type_map[args.flag]
border1s = [0, num_train - args.seq_len, len(df_raw) - num_test - args.seq_len]
border2s = [num_train, num_train + num_vali, len(df_raw)]
border1 = border1s[set_type]
border2 = border2s[set_type]


if set_type == 0:
    border2 = (border2 - args.seq_len) * args.percent // 100 + args.seq_len


if args.features == "M" or args.features == "MS":
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
elif args.features == "S":
    df_data = df_raw[[args.target]]


if args.scale:
    train_data = df_data[border1s[0] : border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)
else:
    data = df_data.values


data = data[border1:border2] 
T, enc_in = data.shape
tot_len = T - args.seq_len - args.pred_len + 1 

def build_prompt(x_seq):       
    min_v, max_v, med = x_seq.min(), x_seq.max(), np.median(x_seq)
    trend = float(np.diff(x_seq).sum())
    lags = topk_lags(x_seq, k=5)   
    prompt = (f"min value {min_v}, max value {max_v}, median value {med}, "
              f"the trend of input is {'upward' if trend>0 else 'downward'}, "
              f"top 5 lags are : {lags.tolist()}")
    return prompt

def topk_lags(x, k=5):
    from scipy.stats import pearsonr
    cors = []
    for lag in range(1, min(len(x)//2, 100)):
        cors.append((lag, pearsonr(x[:-lag], x[lag:])[0]))
    cors = sorted(cors, key=lambda x: abs(x[1]), reverse=True)[:k]
    return np.array([lag for lag,_ in cors])

gpt2.to(args.device)
gpt2.eval()

with torch.no_grad():
    for t in range(tot_len):                 
        seq = data[t:t+args.seq_len]        
        for n in range(enc_in):              
            idx = t * enc_in + n             
            prompt_str = build_prompt(seq[:, n])

            tokens = tokenizer(prompt_str, return_tensors="pt",
                               padding=True, truncation=True, max_length=2048).to(args.device)
            hidden = gpt2(**tokens).last_hidden_state   
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()   

   
            file_path = os.path.join(save_embed_path, f"{idx}.h5")
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("embeddings", data=emb, compression="gzip")

        if t % 100 == 0:
            print(f"[{args.flag}] processed window {t}/{tot_len}")

print(f"All embeddings saved to {save_embed_path}")