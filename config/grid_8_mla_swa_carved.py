# Grid cell 8_mla_swa_carved -- cell 4 with the RoPE channel CARVED out of the head
# instead of added to it (STEP 0b, rope_mode='carved'). Everything else is grid_base.py.
#
# What this cell is for: cell 4 pays a padded score width (48 -> 64 on flex) because
# d_qk = qk_nope + qk_rope is neither a power of two nor equal to d_v. Carving makes
# d_qk = d_v = head_dim = 32, which removes the padding entirely -- at the cost of 8 of
# the 32 content channels per head. The question this cell answers is whether that costs
# quality, and it is the only way to find out: the KV cache barely moves (264 vs 272
# elements/token/layer) and the parameter count drops 2.63%, so the cell runs at a
# DISADVANTAGE on both axes the grid controls for. A tie is therefore a win for carving.
exec(open('config/grid_base.py').read())
attn_type = 'mla'
attn_pattern = 'LLLG'
rope_mode = 'carved'
wandb_run_name = '8_mla_swa_carved'
