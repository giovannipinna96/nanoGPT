# Grid cell 7_mla_all_local -- the other reading of "hybrid".
#
# "SWA + MLA" can be read as an interleaved pattern (cell 4) or as sliding-window
# attention on EVERY layer. Running both turns "you interpreted the task the way that
# suited you" into "I tested both readings", at the cost of one config file.
#
# force_last_global is off here on purpose: resolve_pattern normally forces the last layer
# global, which would make this cell a 7-local/1-global variant of cell 4
# rather than the all-local reading it is meant to be. The difference cell_7 - cell_4 is
# what prices the global layers, which is the central claim of the report.
exec(open('config/grid_base.py').read())
attn_type = 'mla'
attn_pattern = 'L'
force_last_global = False
wandb_run_name = '7_mla_all_local'
