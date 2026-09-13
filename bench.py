"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
import model as model_module
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
# model size (was hardcoded below; needed so the sweep can use the grid architecture)
n_layer = 12
n_head = 12
n_embd = 768
# hybrid attention axes -- must exist as globals for configurator.py to write on them (N1)
attn_type = 'mha'
n_kv_head = None
attn_pattern = 'G'
window_size = 256
force_last_global = True
kv_lora_rank = None
q_lora_rank = None
qk_nope_head_dim = None
qk_rope_head_dim = None
v_head_dim = None
# The four MLA axes below are declared here only so that this benchmark can measure the
# carved cells (8 and 9) with the same tool as the others. No run_all.sh target sets them
# today; without them configurator.py rejects --rope_mode=carved and the training
# throughput of those cells cannot be measured at all. Every default reproduces the
# recorded behaviour exactly, so T2.3 and T5.4 are unaffected.
rope_mode = 'additive'          # MLA only: 'additive' (the grid) | 'carved' | 'reconstructed'
absorb_mode = 'auto'            # MLA decode path; training is always naive, so inert here
symmetric_head_dims = False     # MLA only: d_v = d_qk. Changes the parameter count
mla_up_init = 'bottleneck'      # MLA only: 'bottleneck' (every recorded run) | 'matched'
pos_encoding = 'rope'
rope_theta = 10000.0
attn_impl = 'flex'
csv_out = ''   # append one machine-readable row here
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size = block_size, # how far back does the model look? i.e. context size
    n_layer = n_layer, n_head = n_head, n_embd = n_embd, # size of the model
    dropout = 0, # for determinism
    bias = bias,
    attn_type = attn_type, n_kv_head = n_kv_head,
    attn_pattern = attn_pattern, window_size = window_size, force_last_global = force_last_global,
    kv_lora_rank = kv_lora_rank, q_lora_rank = q_lora_rank,
    qk_nope_head_dim = qk_nope_head_dim, qk_rope_head_dim = qk_rope_head_dim,
    v_head_dim = v_head_dim,
    # Passed, not just declared: a global that never reaches GPTConfig would make
    # --rope_mode=carved succeed silently and benchmark the additive cell instead.
    rope_mode = rope_mode, absorb_mode = absorb_mode,
    symmetric_head_dims = symmetric_head_dims, mla_up_init = mla_up_init,
    pos_encoding = pos_encoding, rope_theta = rope_theta, attn_impl = attn_impl,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    # 20 burn-in steps, not upstream's 10. Raised at some point during the T5.4 sweep
    # (commit d326efc); the reason was not recorded and is not reconstructible from the
    # code or the logs. results/T5.4_training_throughput.csv was produced with 20, so
    # lowering it back would need that table re-measured.
    for stage, num_steps in enumerate([20, 20]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            ms = dt / num_steps * 1000
            peak = torch.cuda.max_memory_allocated() / 1024**3 if device_type == 'cuda' else 0.0
            print(f"time per iteration: {ms:.4f}ms, MFU: {mfu*100:.2f}%")
            print(f"RESULT attn_type={attn_type} pattern={attn_pattern} window={window_size} "
                  f"impl={attn_impl} T={block_size} B={batch_size} ms={ms:.4f} "
                  f"mfu={mfu*100:.2f} peak_gb={peak:.3f} mask_builds={model_module.mask_build_count()}")
            if csv_out:
                import csv as _csv
                new_file = not os.path.exists(csv_out)
                with open(csv_out, 'a', newline='') as fh:
                    w = _csv.writer(fh)
                    if new_file:
                        w.writerow(['attn_type', 'attn_pattern', 'window_size', 'attn_impl',
                                    'block_size', 'batch_size', 'ms_per_iter', 'mfu_pct',
                                    'peak_mem_gb', 'mask_builds'])
                    w.writerow([attn_type, attn_pattern, window_size, attn_impl, block_size,
                                batch_size, f"{ms:.4f}", f"{mfu*100:.2f}", f"{peak:.3f}",
                                model_module.mask_build_count()])
