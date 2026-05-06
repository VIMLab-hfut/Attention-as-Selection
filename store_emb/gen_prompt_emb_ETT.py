#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import h5py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
parser.add_argument("--data_path", type=str, default="ETTh1")
parser.add_argument("--flag", type=str, choices=["train","val","test"], default="val")
parser.add_argument("--save_dir", type=str, default="./Embeddings")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--pred_len", type=int, default=96)
parser.add_argument("--patch_len", type=int, default=16)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--llm_dim", type=int, default=768)   
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


data_name = args.data_path.upper()
if "ETTM" in data_name: 
    BORDER1S = [0,
                12*30*24*4 - args.seq_len,
                12*30*24*4 + 4*30*24*4 - args.seq_len]
    BORDER2S = [12*30*24*4,
                12*30*24*4 + 4*30*24*4,
                12*30*24*4 + 8*30*24*4]
else:  
    BORDER1S = [0,
                12*30*24 - args.seq_len,
                12*30*24 + 4*30*24 - args.seq_len]
    BORDER2S = [12*30*24,
                12*30*24 + 4*30*24,
                12*30*24 + 8*30*24]

data_path_file = args.data_path.replace(".csv", "")
os.makedirs(os.path.join(args.save_dir, data_path_file, args.flag), exist_ok=True)


config = GPT2Config.from_pretrained("openai-community/gpt2")
gpt2 = GPT2Model.from_pretrained("openai-community/gpt2", config=config).to(args.device)
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2.eval()
for p in gpt2.parameters():
    p.requires_grad = False


df_raw = pd.read_csv(os.path.join(args.root_path, data_path_file + ".csv"))
border1, border2 = BORDER1S[{"train":0,"val":1,"test":2}[args.flag]], BORDER2S[{"train":0,"val":1,"test":2}[args.flag]]
df_data = df_raw[df_raw.columns[1:]]          
scaler = StandardScaler()
scaler.fit(df_data.values[BORDER1S[0]:BORDER2S[0]])   
data = scaler.transform(df_data.values[border1:border2])  
T, enc_in = data.shape
tot_len = T - args.seq_len - args.pred_len + 1

def build_prompt(x_seq):          # x_seq: (seq_len,)
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

            file_path = os.path.join(args.save_dir, data_path_file, args.flag, f"{idx}.h5")
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("embeddings", data=emb, compression="gzip")

        if t % 100 == 0:
            print(f"[{args.flag}] processed window {t}/{tot_len}")

print("All embeddings saved.")
