import torch
import torch.nn as nn
import torch.nn.functional as F

class AblatableSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, layer_idx, ablation_mask):
        B, T, C = x.size()
        
        # Calculate Query, Key, Value matrices
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # QK^T Circuit for raw attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Apply causal mask (prevent looking into the future)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # INTERVENTION: Zero out specific heads based on the mask
        if ablation_mask is not None and layer_idx in ablation_mask:
            for head_idx in ablation_mask[layer_idx]:
                attn_probs[:, head_idx, :, :] = 0.0
                
        # OV Circuit composition
        out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), attn_probs

class AttentionOnlyTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = AblatableSelfAttention(embed_dim, num_heads)
        
    def forward(self, x, layer_idx, ablation_mask):
        # The stream is only read by attention, and the output is added linearly back.
        attn_out, attn_probs = self.attn(self.ln_1(x), layer_idx, ablation_mask)
        x = x + attn_out 
        return x, attn_probs

class AttentionOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(1024, embed_dim) # Max seq length 1024
        
        self.blocks = nn.ModuleList([
            AttentionOnlyTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx, ablation_mask=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        
        all_attention_matrices = []
        for layer_idx, block in enumerate(self.blocks):
            x, attn_probs = block(x, layer_idx, ablation_mask)
            all_attention_matrices.append(attn_probs)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, all_attention_matrices