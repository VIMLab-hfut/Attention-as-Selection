
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Cross_Modal_Align import CrossModalStable
from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2Tokenizer,
)
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize


transformers.logging.set_verbosity_error()


class LearnableTemperatureInfoNCELoss(nn.Module):
    def __init__(self, initial_temperature=0.07):
        super().__init__()
        # learnbale temperature
        self.log_temperature = nn.Parameter(torch.tensor(initial_temperature).log())

    def forward(self, query, positive, negative=None):
        """
        Compute InfoNCE loss
        Args:
            query: [batch_size, seq_len, dim] - time series encoder outputs
            positive: [batch_size, seq_len, dim] - LLM outputs (positive samples)
            negative: [batch_size, seq_len, dim] - other time steps' LLM outputs (negative samples)
        """

        temperature = self.log_temperature.exp()

        # collapse sequence dim by mean similar to original code
        query = query.mean(dim=1)
        positive = positive.mean(dim=1)
        if negative is None:
            negative = torch.roll(positive, shifts=1, dims=0)
        query = F.normalize(query, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        pos_logits = torch.sum(query * positive, dim=-1) / temperature
        neg_logits = torch.matmul(query, negative.t()) / temperature
        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.seed = configs.seed
        self.data = configs.data
        self.llm_model = configs.llm_model
        self.fusion_ln = nn.LayerNorm(self.d_llm)
        if configs.llm_model == "GPT2":
            self.gpt2_config = GPT2Config.from_pretrained("openai-community/gpt2")

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except (
                EnvironmentError
            ):  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    local_files_only=False,
                )
        
        else:
            raise Exception("LLM model is not defined")

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False
            # param.requires_grad = True

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout 
        )

        self.backcast_len = configs.seq_len
        # mask
        self.mask_rate = configs.mask_rate

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.prompt_len = min(
            self.patch_nums, int(self.patch_nums * configs.pred_len / configs.seq_len)
        )
        self.head_nf = self.d_ff * (self.prompt_len + self.patch_nums)

        if (
            self.task_name == "long_term_forecast"
            or self.task_name == "short_term_forecast"
        ):
            if self.mask_rate == 0:
                self.output_projection = FlattenHead(
                    configs.enc_in,
                    self.head_nf,
                    self.pred_len,
                    head_dropout=configs.dropout,
                )
            else:
                self.output_projection = FlattenHead(
                    configs.enc_in,
                    self.head_nf,
                    self.pred_len + self.backcast_len,
                    head_dropout=configs.dropout,
                )
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.tsprojection = nn.Linear(configs.d_model, configs.llm_dim)

        self.alignment_weight = 0.5

        # learnable prompt
        self.learnable_prompt = nn.Parameter(torch.randn(self.prompt_len, self.d_llm))

        self.LearnableTemperatureInfoNCELoss = LearnableTemperatureInfoNCELoss(
            initial_temperature=0.07
        )
        
        self.prompt_expander = nn.Linear(768, self.prompt_len * self.d_llm)   # 768 → prompt_token*hidden
        self.prompt_mlp = nn.Sequential(
            nn.Linear(self.d_llm, self.d_llm),
            nn.ReLU(),
            nn.Linear(self.d_llm, self.d_llm)
        )


  
        self.cross_t2s = CrossModalStable(d_model=768, n_heads=8)
        self.cross_s2t = CrossModalStable(d_model=768, n_heads=8)

        self.cross_ln_text = nn.LayerNorm(self.d_llm)
        self.cross_ln_time = nn.LayerNorm(self.d_llm)

  
        self.cross_gamma = nn.Parameter(torch.tensor(0.01))
        self.gamma = nn.Parameter(torch.tensor(self.seq_len / 10.0))



    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        *,
        embeddings=None,
        epoch=None,
        batch=None,
        mode="train",
        batch_y=None
    ):

        dec_out, alignment_loss = self.forecast(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
            x_dec=x_dec,
            x_mark_dec=x_mark_dec,
            embeddings=embeddings,
            batch_y=batch_y,
            epoch=epoch,
            mode=mode
        )
        return dec_out, alignment_loss

    def forecast(  
        self,
        x_enc,#inputdata
        x_mark_enc,
        x_dec,
        x_mark_dec,
        embeddings,      
        batch_y=None,
        epoch=None,
        mode='train'
    ):
        x_enc = self.normalize_layers(x_enc, "norm")  
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))          
        prompt_embeddings = embeddings.to(x_enc.device)                       

        prompt_embeddings = prompt_embeddings.unsqueeze(1).repeat(1, N, 1)  
        prompt_embeddings = prompt_embeddings.reshape(B * N, -1)           

        B_N = prompt_embeddings.size(0)

        prompt_seq = self.prompt_expander(prompt_embeddings)                 
        prompt_seq = prompt_seq.view(B_N, self.prompt_len, self.d_llm)    
        Learn_prompt = self.learnable_prompt.unsqueeze(0).expand(B_N, -1, -1)
        prompt_embeddings_all = torch.cat([Learn_prompt, prompt_seq], dim=1)
        TS = enc_out
        TS = self.tsprojection(TS)
        llama_inputs = prompt_embeddings_all 
        llm_out = self.llm_model(inputs_embeds=llama_inputs)
        llm_hidden_states = llm_out.last_hidden_state   

        dec_out = llm_hidden_states[:, -self.prompt_len :, :]  
        time_attn_out, attn = self.cross_t2s(TS, dec_out)  
        def kl_to_low_entropy_prior(w, alpha=4.0, eps=1e-8):
            """
            w: [B, T], normalized attention weights
            Encourage w to be close to its sharpened (low-entropy) version
            """
            # ---- clamp to avoid log(0) ----
            w_safe = torch.clamp(w, eps, 1.0)

            with torch.no_grad():
                p = w_safe.pow(alpha)
                p = p / p.sum(dim=-1, keepdim=True)

            kl = (w_safe * (w_safe.log() - p.log())).sum(dim=-1)
            return kl.mean()
        TS_fused = TS * (1.0 + self.gamma * time_attn_out.unsqueeze(-1))

        infonce_loss = self.LearnableTemperatureInfoNCELoss(TS_fused, dec_out)
        kl_loss = kl_to_low_entropy_prior(time_attn_out, alpha=4.0)

        alignment_loss = infonce_loss + self.cross_gamma * kl_loss


        dec_out = torch.cat(
            (dec_out[:, :, : self.d_ff], TS_fused[:, :, : self.d_ff]), dim=1
        )

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, "denorm")
     
        # print(alignment_loss)
        return dec_out, alignment_loss
        