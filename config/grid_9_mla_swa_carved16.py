# Grid cell 9_mla_swa_carved16 -- the SINGLE-VARIABLE version of the carving question.
#
# Cell 8 changes two things at once against cell 4: the content width (32 -> 24) AND the
# positional channel (d_rope 16 -> 8, i.e. 8 frequency pairs -> 4). If it loses, the loss
# cannot be attributed. This cell keeps d_rope at 16, exactly as cell 4:
#
#            d_nope  d_rope  d_qk  cache/tok  params
#   cell 4      32      16     48      272    50.998M   (additive, padded 48 -> 64 on flex)
#   cell 8      24       8     32      264    49.654M   (-2.63%)
#   cell 9      16      16     32      272    49.425M   (-3.08%)
#
# so against cell 4 this cell has the SAME positional channel and the SAME KV cache to the
# byte, and exactly one thing differs: the content width, halved. That is the carving
# question in isolation, and it is the harsher cut of the two -- if -50% of content
# survives, the -25% of cell 8 is safe a fortiori.
#
# Note the structural limit this exposes, worth one line in the report: at head_dim=32
# carving has no free parameters, because d_nope + d_rope = head_dim forces content and
# position to trade against each other directly. At head_dim=64 it does: d_rope=16 leaves
# d_nope=48, a 25% cut instead of 50%. Another reason the prize is the 8-head config.
exec(open('config/grid_base.py').read())
attn_type = 'mla'
attn_pattern = 'LLLG'
rope_mode = 'carved'
qk_rope_head_dim = 16            # NOT the carved default (8): matched to cell 4
wandb_run_name = '9_mla_swa_carved16'
