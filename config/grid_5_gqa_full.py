# Grid cell 5_gqa_full -- the ISO-CACHE control, not an iso-parameter one.
#
# considerazioni_finali.md 2.1: the DeepSeek paper does not argue MLA against MHA, it
# argues it against GQA. Without this cell the obvious objection -- "GQA cuts the cache by
# as much in three lines of code, so why MLA?" -- has no answer in the report.
#
# On configuration A (n_head=16, head_dim=32) the iso-cache point is n_kv_head=4:
#   GQA-4 cache/token/layer = 2 * 4 * 32          = 256 elements
#   MLA   cache/token/layer = d_c + qk_rope = 256 + 16 = 272 elements
# so GQA has SLIGHTLY LESS cache than MLA, exactly as in the documents' configuration.
# It also has ~12.5% FEWER parameters by construction, which is why gate T3.5 excludes it
# from the +/-2% criterion: this cell is matched on memory, not on capacity, and the
# report must say so wherever its loss is quoted.
exec(open('config/grid_base.py').read())
attn_type = 'gqa'
n_kv_head = 4
attn_pattern = 'G'
wandb_run_name = '5_gqa_full'
