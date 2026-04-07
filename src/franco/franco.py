import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings as RotaryEmBEDDIES

logger = logging.getLogger(__name__)

class EmbebeddiesLayers(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model) # this is a lookup table practically, it will be trained to learn the embeddings for each token in the vocab
        self.dropout = nn.Dropout(dropout) # generalization 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()

        assert d_model % n_heads == 0 # the heads must be a divisor of d_models
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.rope  = RotaryEmBEDDIES(
            dim=self.head_dim,
            max_seq_len=max_seq_len
        )

        # ATTENTION

        self.norm1 = nn.RMSNorm(d_model, eps = eps)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # FEED THE FORWARD NETWORK

        self.norm2 = nn.RMSNorm(d_model, eps = eps)
        self.W1    = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.W3    = nn.Linear(d_model, d_ff, bias=False)   # hidden
        self.W2    = nn.Linear(d_ff, d_model, bias=False)   # proj out

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C = x.shape

        # -- attention --

        x_norm = self.norm1(x)
        
        Q = self.W_Q(x_norm).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        K = self.W_K(x_norm).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        V = self.W_V(x_norm).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        Q = self.rope(Q)
        K = self.rope(K)

        # I need to see what is nn.functional
        attn_out = F.scaled_dot_product_attention(Q,K,V, 
                                                  is_causal = True, 
                                                  dropout_p = 
                                                     self.dropout.p 
                                                  if self.training else 0.0)

        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.W_O(attn_out)

        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        x_ff = (F.silu(self.W1(x_norm)) * self.W3(x_norm))
        x_ff = self.W2(x_ff)

        x = x + self.dropout(x_ff)

        return x

class FRANCO(nn.Module):
    def __init__(self, vocab_size,
                       d_model, 
                       n_layers, 
                       n_head, 
                       d_ff, 
                       eps_rms_norm, 
                       dropout, 
                       seq_len = 512,
                       mods = "vanilla" # this is for future use for attention and ff variants
                ):
        
        super().__init__()

        self.embedding = EmbebeddiesLayers(vocab_size, d_model, dropout)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, seq_len,  dropout, eps_rms_norm)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.RMSNorm(d_model, eps = eps_rms_norm)
        self.lm_head = nn.linear(d_model, vocab_size, bias = False)
        self.lm_head.weight = self.embedding.embed.weight


    def forward(self, idx, targets = None):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss



        











        

