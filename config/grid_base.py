# Shared settings for the 2x2 grid (phase G, reference configuration), on the adopted
# configuration A. Every cell imports this and overrides
# only its attention axes, so nothing but the attention can differ between cells.
#
# Deviation from the initial design, declared: it had n_head=8. Configuration
# A uses n_head=16 (head_dim=32), adopted because head_dim=64 gives
# d_qk = 64+32 = 96 -> padded to 128, whose BACKWARD does not compile through FlexAttention
# on sm80 (gates T3.4/T3.6). Every measurement of this campaign is on
# configuration A, so the grid must be too.

out_dir = 'out/grid'
eval_interval = 250
eval_iters = 200          # only used for the train-split diagnostic; val uses VAL_SET
log_interval = 10
always_save_checkpoint = False   # keep the best, not the last

dataset = 'finewebedu'
gradient_accumulation_steps = 20
batch_size = 24                  # 20 * 24 * 1024 = 491,520 tokens/iter
block_size = 1024

# configuration A
n_layer = 8
n_head = 16
n_embd = 512
dropout = 0.0
bias = False
pos_encoding = 'rope'            # RoPE in EVERY cell: the grid isolates attention, not
                                 # the position encoding (P3)
rope_theta = 10000.0
attn_impl = 'flex'               # flex for every grid cell; sdpa_mask stays in the tests
kv_lora_rank = 256               # 8 * head_dim, set by the iso-parameter gate T3.5
window_size = 256

learning_rate = 6e-4
max_iters = 2000                 # ~983M tokens, ~0.65 epoch of the FineWeb-Edu shard
lr_decay_iters = 2000
warmup_iters = 200
min_lr = 6e-5
