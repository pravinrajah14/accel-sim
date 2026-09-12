"""Real layer shapes -- as data.

Pulled from a GPT-2 small transformer block (the shapes people actually
train).  One block, batch=8, sequence=1024  ->  M = batch * seq = 8192 tokens.

  d_model = 768        d_ff = 3072        n_heads = 12        d_head = 64

GEMMs in a block:

  q_proj    X(M,768)  @ W(768,768)
  k_proj    X(M,768)  @ W(768,768)
  v_proj    X(M,768)  @ W(768,768)
  attn_out  X(M,768)  @ W(768,768)
  mlp_up    X(M,768)  @ W(768,3072)
  mlp_down  X(M,3072) @ W(3072,768)

Plus the attention op itself (`simulator/attention.py`) -- the QK^T ->
softmax -> (softmax)@V core, batch=8, seq=1024, 12 heads x 64 = 768.
`GPT2_BLOCK` is the 6 GEMMs alone (v1 scope); `GPT2_BLOCK_FULL` adds
attention for a complete transformer layer.
"""

from .attention import Attention
from .dataflow import Layer

BATCH = 8
SEQ = 1024
M = BATCH * SEQ          # 8192 tokens
D_MODEL = 768
D_FF = 3072
N_HEADS = 12
D_HEAD = D_MODEL // N_HEADS   # 64

GPT2_BLOCK = [
    Layer("q_proj",   M=M, K=D_MODEL, N=D_MODEL),
    Layer("k_proj",   M=M, K=D_MODEL, N=D_MODEL),
    Layer("v_proj",   M=M, K=D_MODEL, N=D_MODEL),
    Layer("attn_out", M=M, K=D_MODEL, N=D_MODEL),
    Layer("mlp_up",   M=M, K=D_MODEL, N=D_FF),
    Layer("mlp_down", M=M, K=D_FF,    N=D_MODEL),
]

ATTENTION = Attention("attention", batch=BATCH, seq=SEQ,
                      n_heads=N_HEADS, d_head=D_HEAD)

GPT2_BLOCK_FULL = GPT2_BLOCK + [ATTENTION]

# De-duplicated distinct shapes (q/k/v/attn_out are identical) for validation
# runs where each unique shape only needs to go through Timeloop once.
DISTINCT_SHAPES = [
    Layer("proj_768x768",   M=M, K=D_MODEL, N=D_MODEL),
    Layer("mlp_up_768x3072", M=M, K=D_MODEL, N=D_FF),
    Layer("mlp_down_3072x768", M=M, K=D_FF, N=D_MODEL),
]
