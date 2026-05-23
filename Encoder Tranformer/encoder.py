import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
 
# ───────────────────────────────────────────────────────────────── 
# SCALED DOT-PRODUCT ATTENTION 
# ───────────────────────────────────────────────────────────────── 
class ScaledDotProductAttention(nn.Module): 
    """ 
    Computes: Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V 
 
    Args: 
        dropout: applied to the attention weights during training 
    """ 
    def __init__(self, dropout=0.1): 
        super().__init__() 
        self.dropout = nn.Dropout(dropout) 
 
    def forward(self, Q, K, V, mask=None): 
        d_k = Q.size(-1)  # dimension per head 
 
        # Compute raw attention scores: (B, h, seq_q, seq_k) 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) 
 
        # Apply optional mask (used for padding or causal masking) 
        if mask is not None: 
            # Set masked positions to -infinity so softmax gives 0 
            scores = scores.masked_fill(mask == 0, -1e9) 
 
        # Softmax over the key dimension → attention weights 
        attn_weights = F.softmax(scores, dim=-1) 
        attn_weights = self.dropout(attn_weights) 
 
        # Weighted sum of values 
        output = torch.matmul(attn_weights, V)  # (B, h, seq_q, d_v) 
        return output, attn_weights 
 
 
# ───────────────────────────────────────────────────────────────── 
# MULTI-HEAD ATTENTION 
# ───────────────────────────────────────────────────────────────── 
class MultiHeadAttention(nn.Module): 
    """ 
    Runs h=8 attention heads in parallel, then projects back to 
d_model. 
 
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O 
    head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V) 
    """ 
    def __init__(self, d_model=512, h=8, dropout=0.1): 
        super().__init__() 
        assert d_model % h == 0, 'd_model must be divisible by h' 
        self.h      = h 
        self.d_k    = d_model // h        # 64 per head 
        self.d_model = d_model 
 
        # Four linear projections: Q, K, V, and output 
        self.W_q = nn.Linear(d_model, d_model, bias=False) 
        self.W_k = nn.Linear(d_model, d_model, bias=False) 
        self.W_v = nn.Linear(d_model, d_model, bias=False) 
        self.W_o = nn.Linear(d_model, d_model, bias=False) 
 
        self.attn    = ScaledDotProductAttention(dropout) 
        self.dropout = nn.Dropout(dropout) 
 
    def _split_heads(self, x, batch_size): 
        """Reshape (B, seq, d_model) → (B, h, seq, d_k)""" 
        x = x.view(batch_size, -1, self.h, self.d_k) 
        return x.transpose(1, 2)     # (B, h, seq, d_k) 
 
    def forward(self, Q, K, V, mask=None): 
        B = Q.size(0)  # batch size 
 
        # Linear projections → split into heads 
        Q = self._split_heads(self.W_q(Q), B)   # (B, h, seq_q, d_k) 
        K = self._split_heads(self.W_k(K), B)   # (B, h, seq_k, d_k) 
        V = self._split_heads(self.W_v(V), B)   # (B, h, seq_k, d_k) 
 
        # Run attention in parallel across all heads 
        x, self.attn_weights = self.attn(Q, K, V, mask)  # (B, h, 
seq_q, d_k) 
 
        # Concatenate heads: (B, h, seq, d_k) → (B, seq, d_model) 
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model) 
 
        # Final output projection 
        return self.W_o(x)    # (B, seq_q, d_model) 
 
 
# ───────────────────────────────────────────────────────────────── 
# POSITION-WISE FEED-FORWARD NETWORK 
# ───────────────────────────────────────────────────────────────── 
class PositionwiseFFN(nn.Module): 
    """ 
    FFN(x) = ReLU(x·W_1 + b_1) · W_2 + b_2 
 
    Applied independently to each position. 
    Inner dimension d_ff is typically 4 × d_model = 2048. 
    """ 
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1): 
        super().__init__() 
        self.linear1  = nn.Linear(d_model, d_ff) 
        self.linear2  = nn.Linear(d_ff, d_model) 
        self.dropout  = nn.Dropout(dropout) 
        self.relu     = nn.ReLU() 
 
    def forward(self, x): 
        # Expand → non-linearity → contract 
        return self.linear2(self.dropout(self.relu(self.linear1(x)))) 
 
 
# ───────────────────────────────────────────────────────────────── 
# ENCODER LAYER (one of N=6 identical layers) 
# ───────────────────────────────────────────────────────────────── 
class EncoderLayer(nn.Module): 
    """ 
    Single encoder layer with two sub-layers: 
      1. Multi-Head Self-Attention + Add & Norm 
      2. Position-wise FFN         + Add & Norm 
    """ 
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1): 
        super().__init__() 
        self.self_attn = MultiHeadAttention(d_model, h, dropout) 
        self.ffn       = PositionwiseFFN(d_model, d_ff, dropout) 
        self.norm1     = nn.LayerNorm(d_model) 
        self.norm2     = nn.LayerNorm(d_model) 
        self.dropout   = nn.Dropout(dropout) 
 
    def forward(self, x, src_mask=None): 
        # Sub-layer 1: Self-attention with residual + LayerNorm 
        _attn = self.self_attn(x, x, x, src_mask)   # Q=K=V=x (self) 
        x = self.norm1(x + self.dropout(_attn)) 
 
        # Sub-layer 2: FFN with residual + LayerNorm 
        _ffn = self.ffn(x) 
        x = self.norm2(x + self.dropout(_ffn)) 
        return x                                      # (B, seq, 
d_model) 
 
 
 
# ───────────────────────────────────────────────────────────────── 
# FULL ENCODER STACK 
# ───────────────────────────────────────────────────────────────── 
class Encoder(nn.Module): 
    """ 
    Full encoder: embedding + positional encoding + N encoder layers. 
 
    Input:  token indices (B, seq_len) 
    Output: contextual representations (B, seq_len, d_model) 
    """ 
    def __init__(self, vocab_size, d_model=512, N=6, h=8, 
                 d_ff=2048, max_len=5000, dropout=0.1): 
        super().__init__() 
        self.d_model   = d_model 
        self.embed     = nn.Embedding(vocab_size, d_model) 
        self.layers    = nn.ModuleList( 
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)] 
        ) 
        self.norm      = nn.LayerNorm(d_model) 
        self.dropout   = nn.Dropout(dropout) 
 
        # Pre-compute sinusoidal positional encodings once 
        self.register_buffer('pe', self._build_pe(max_len, d_model)) 
 
    @staticmethod 
    def _build_pe(max_len, d_model): 
        """Build sinusoidal positional encoding table (max_len, 
d_model).""" 
        pe  = torch.zeros(max_len, d_model) 
        pos = torch.arange(0, max_len).unsqueeze(1).float() 
        # Divide term for each dimension pair 
        div = torch.exp( 
            torch.arange(0, d_model, 2).float() 
            * (-math.log(10000.0) / d_model) 
        ) 
        pe[:, 0::2] = torch.sin(pos * div)    # even dims 
        pe[:, 1::2] = torch.cos(pos * div)    # odd dims 
        return pe.unsqueeze(0)                 # (1, max_len, d_model) 
 
    def forward(self, src, src_mask=None): 
        # Step 1: Embedding lookup + scale 
        x = self.embed(src) * math.sqrt(self.d_model)  # (B, seq, 
d_model) 
 
        # Step 2: Add positional encoding 
        seq_len = src.size(1) 
        x = x + self.pe[:, :seq_len, :]                # broadcast over 
batch 
 
        # Step 3: Apply dropout 
        x = self.dropout(x) 
 
        # Step 4: Pass through N encoder layers 
        for layer in self.layers: 
            x = layer(x, src_mask) 
 
        # Step 5: Final layer normalisation 
        return self.norm(x)                             # (B, seq, 
d_model) 
 
 
# ───────────────────────────────────────────────────────────────── 
# USAGE EXAMPLE: Classification head on top of encoder 
# ───────────────────────────────────────────────────────────────── 
class EncoderForClassification(nn.Module): 
    """Encoder + linear classifier (e.g., sentiment analysis).""" 
    def __init__(self, vocab_size, num_classes, d_model=512): 
        super().__init__() 
        self.encoder    = Encoder(vocab_size, d_model) 
        self.classifier = nn.Linear(d_model, num_classes) 
 
    def forward(self, src, src_mask=None): 
        enc_out = self.encoder(src, src_mask)  # (B, seq, d_model) 
        # Use the [CLS] token representation (position 0) for 
classification 
        cls_repr = enc_out[:, 0, :]             # (B, d_model) 
        logits   = self.classifier(cls_repr)    # (B, num_classes) 
        return logits 
# Example usage: 
# model = EncoderForClassification(vocab_size=30522, num_classes=2) 


# ───────────────────────────────────────────────────────────────── 
# Padding Mask 
# ───────────────────────────────────────────────────────────────── 
def make_padding_mask(src, pad_idx=0): 
""" 
Creates a padding mask of shape (B, 1, 1, seq_len). 
Value is 1 where token is real, 0 where token is padding. 
The extra dimensions broadcast correctly with (B, h, seq, seq) 
scores. 
""" 
return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq) 
# Usage: 
# src = torch.tensor([[101, 2054, 2003, 102, 0, 0]])  # 0 = [PAD] 
# mask = make_padding_mask(src, pad_idx=0) 
# enc_out = encoder(src, mask)
# logits = model(token_ids, padding_mask)  # binary sentiment
