
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class CrossAttentionSelector(nn.Module):
    """
    Attention-as-Selector
    """
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        self.kv_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # ===== analysis-only buffer =====
        self.latest_attn = None
        self.latest_attn_entropy = None
        # ===== analysis cache =====
        self.save_analysis = False
        self.analysis_dir = None
        self.step_counter = 0

    def forward(self, query, context, mask=None):

        q = self.q_proj(query)
        kv = self.kv_proj(context)
        _, attn = self.attn(
            q,
            kv,
            kv,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False
        )
        # attn: [B, H, T, P]

        # ===== cache =====
        self.latest_attn = attn.detach()


        time_weight = attn.mean(dim=1).mean(dim=-1)   # [B, T]
        time_weight = time_weight / (time_weight.sum(dim=-1, keepdim=True) + 1e-6)

        entropy = -(time_weight * (time_weight + 1e-8).log()).sum(dim=-1)
        self.latest_attn_entropy = entropy.detach()
        # print("raw_signal",raw_signal)
        # if self.save_analysis and self.analysis_dir is not None:
        #     os.makedirs(self.analysis_dir, exist_ok=True)

        #     save_dict = {
        #         "attn": attn.detach().cpu(),               # [B,H,T,P]
        #         "time_weight": time_weight.detach().cpu(), # [B,T]
        #         "entropy": entropy.detach().cpu(),         # [B]
        #     }

        #     torch.save(
        #         save_dict,
        #         f"{self.analysis_dir}/step_{self.step_counter}.pt"
        #     )
        #     self.step_counter += 1
        return time_weight, attn

class CrossModalStable(nn.Module):
  
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.selector = CrossAttentionSelector(d_model, n_heads, dropout)

    def forward(self, query, context, raw_signal=None,mask=None):
     
        query_norm = self.norm(query)
        time_weight, attn = self.selector(query_norm, context, mask)

        return time_weight, attn

class CrossAttentionQKVFusion(nn.Module):
    """
    Ablation: Standard QKV Cross-Attention

    """
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # analysis-only
        self.latest_attn = None

    def forward(self, query, context, mask=None):
        """
        query   : [B, T, D]  (time)
        context : [B, P, D]  (text / prompt)
        """
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)


        fused_feat, attn = self.attn(
            q, k, v,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=False
        )
        # fused_feat: [B, T, D]

        self.latest_attn = attn.detach()

        return fused_feat, attn

class CrossModalFusion(nn.Module):
    """
    Ablation: Cross-modal feature fusion
    """
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttentionQKVFusion(
            d_model, n_heads, dropout
        )

    def forward(self, query, context, mask=None):
        """
        return fused temporal features
        """
        query_norm = self.norm(query)
        fused_feat, attn = self.cross_attn(query_norm, context, mask)
        return fused_feat, attn
