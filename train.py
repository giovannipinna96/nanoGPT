"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
# --- fixed validation set (gate T0.3) ---
# nanoGPT's estimate_loss draws a FRESH random batch at every evaluation, so the val loss
# it prints carries a sampling variance on top of the seed variance. The differences this
# project is looking for are 0.01-0.05 nats, the same order as that noise: left as is, the
# grid would be comparing dataloaders, not architectures. val_batches batches are drawn
# ONCE from a dedicated generator and reused by every cell and every seed.
val_batches = 200         # 0 -> keep nanoGPT's random sampling
val_seed = 1234           # fixed across cells AND seeds: it selects the measuring stick
# Training seed. nanoGPT hardwires 1337; gate T0.3 needs the SAME cell trained twice with
# a different one to measure sigma_seed, the significance threshold of the whole grid, and
# a global declared here is the only thing configurator.py will write on (N1).
train_seed = 1337
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# --- hybrid attention (SWA x MLA) ---
# Every one of these MUST be declared here AND appear in model_args below. configurator.py
# raises on an unknown --flag, so a missing declaration fails loudly; but a config file is
# exec'd into these globals, so `window_size = 256` in config/grid_*.py always "works" and
# is silently dropped if the name never reaches GPTConfig. That is threat N1: the run then
# trains with the default window and says nothing.
attn_type = 'mha'         # 'mha' | 'gqa' | 'mla'
n_kv_head = None          # GQA only; None -> n_head
attn_pattern = 'G'        # tiled over layers, e.g. 'LLLG'; last layer forced global
window_size = 256         # W visible tokens INCLUDING self
force_last_global = True  # False only for the all-local cell 7: left on,
                          # attn_pattern='L' resolves to 7 local + 1 global
kv_lora_rank = None       # MLA d_c;   None -> 4 * head_dim
q_lora_rank = None        # MLA d'_c;  None -> no query compression
qk_nope_head_dim = None   # None -> head_dim (full content dim, RoPE part is additive)
qk_rope_head_dim = None   # None -> head_dim // 2
v_head_dim = None         # None -> head_dim
rope_mode = 'additive'    # MLA only: 'additive' (DeepSeek, the grid) | 'carved' (the
                          # channel is taken out of head_dim, so d_qk == d_v) |
                          # 'reconstructed' (no channel; the rebuilt k is rotated)
absorb_mode = 'auto'      # MLA decode path: 'auto' (absorbed below T*) | 'never' |
                          # 'always'. Runtime only -- it changes no weight, and training
                          # always uses the naive form whatever this says
symmetric_head_dims = False  # MLA only: force v_head_dim = qk_nope + qk_rope, so that
                          # q/k/v share one width and impl='flash' needs no padding.
                          # Off by default: it changes the parameter count (STEP 0)
mla_up_init = 'bottleneck'  # MLA only: 'bottleneck' (every recorded run) | 'matched'
                          # (k/v init variance equal to MHA)
pos_encoding = 'learned'  # 'learned' | 'rope'
rope_theta = 10000.0
attn_impl = 'sdpa_mask'   # 'sdpa_mask' (oracle) | 'flex' (grid runs) | 'flash'
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(train_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def build_fixed_val(n_batches, seed):
    """Draw n_batches validation batches ONCE, from a generator of their own.

    The generator is local on purpose: sampling through the global RNG would shift every
    subsequent draw, so the training stream (and, in a from-scratch run, nothing else --
    but the T1.1 oracle depends on that stream being reproducible) would change depending
    on how large the val set is. Batches are kept on the CPU and moved per evaluation:
    200 x 24 x 1024 int64 is ~39 MB per tensor (~79 MB for x and y together), which is not
    worth holding on the device
    where it would perturb the peak-memory numbers of Phase F.
    """
    g = torch.Generator().manual_seed(seed)
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (n_batches, batch_size), generator=g)
    batches = []
    for row in ix:
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in row])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in row])
        batches.append((x, y))
    # Report the size of the measuring stick, and how much of it is actually distinct:
    # the offsets are drawn with replacement from a val.bin of len(data) tokens, so the
    # windows overlap and the nominal token count overstates the independent sample.
    covered = torch.zeros(len(data), dtype=torch.bool)
    for row in ix:
        for i in row.tolist():
            covered[i:i+block_size] = True
    nominal = n_batches * batch_size * block_size
    print(f"fixed val set: {n_batches} batches x {batch_size} x {block_size} = "
          f"{nominal/1e6:.2f}M tokens ({covered.sum().item()/1e6:.2f}M distinct of "
          f"{len(data)/1e6:.2f}M in val.bin), seed={seed}")
    return batches


VAL_SET = build_fixed_val(val_batches, val_seed) if val_batches else None

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  attn_type=attn_type, n_kv_head=n_kv_head, attn_pattern=attn_pattern,
                  window_size=window_size, force_last_global=force_last_global, kv_lora_rank=kv_lora_rank,
                  q_lora_rank=q_lora_rank, qk_nope_head_dim=qk_nope_head_dim,
                  qk_rope_head_dim=qk_rope_head_dim, v_head_dim=v_head_dim,
                  symmetric_head_dims=symmetric_head_dims, mla_up_init=mla_up_init,
                  rope_mode=rope_mode, absorb_mode=absorb_mode,
                  pos_encoding=pos_encoding, rope_theta=rope_theta,
                  attn_impl=attn_impl) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
          'attn_type', 'n_kv_head', 'attn_pattern', 'window_size',
          'force_last_global',
          'kv_lora_rank', 'q_lora_rank', 'qk_nope_head_dim',
          'qk_rope_head_dim', 'v_head_dim', 'symmetric_head_dims', 'rope_mode', 'mla_up_init',
          'pos_encoding', 'rope_theta']:
        # A checkpoint written before a field existed lacks its key: no checkpoint of the
        # campaign has mla_up_init, and the earliest grid runs (cells 1-5) also predate
        # force_last_global, symmetric_head_dims and rope_mode. Indexing a missing key
        # died on a KeyError. Every such default is the only
        # behaviour the code had before the field was added, i.e. the one those runs
        # were trained with, so falling back to it rebuilds the same model.
        model_args[k] = checkpoint_model_args.get(k, GPTConfig.__dataclass_fields__[k].default)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
          'attn_type', 'n_kv_head', 'attn_pattern', 'window_size',
          'force_last_global',
          'kv_lora_rank', 'q_lora_rank', 'qk_nope_head_dim',
          'qk_rope_head_dim', 'v_head_dim', 'symmetric_head_dims', 'rope_mode', 'mla_up_init',
          'pos_encoding', 'rope_theta']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

def _fingerprint(x):
    """Order-sensitive checksum of a batch of token ids.

    Position-weighted rather than a plain sum, so two batches holding the same tokens in
    a different order do not collide. This is what makes "every cell evaluated on the
    same data" a checkable claim instead of an assumption.
    """
    w = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.int64)
    return int((x.reshape(-1).to(torch.int64) * w).sum().item())


_val_fingerprint = None


def _check_val_fingerprint(fp):
    """Print the val fingerprint once, and shout if it ever moves.

    It must be identical across evaluations of one run AND across every cell of the grid.
    With the original random sampling it is neither: architectures consume the global RNG
    differently at init, so each cell would silently validate on different tokens.
    """
    global _val_fingerprint
    if _val_fingerprint is None:
        _val_fingerprint = fp
        print(f"val fingerprint: {fp}")
    elif fp != _val_fingerprint:
        print(f"WARNING val fingerprint changed: {_val_fingerprint} -> {fp} "
              f"(val set is NOT fixed; cross-cell comparison is unsound)")
        _val_fingerprint = fp


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        # 'val' walks the FIXED set built above, identical in every cell and every seed,
        # so the number the grid compares has no sampling noise of its own (T0.3).
        # 'train' stays randomly sampled: it is a diagnostic, and a fixed train subset
        # would slowly turn into a memorisation probe rather than a loss estimate.
        batches = VAL_SET if (split == 'val' and VAL_SET is not None) else None
        n = len(batches) if batches is not None else eval_iters
        losses = torch.zeros(n)
        fingerprint = 0
        for k in range(n):
            if batches is not None:
                X, Y = batches[k]
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            else:
                X, Y = get_batch(split)
            if split == 'val':
                fingerprint += _fingerprint(X)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        if split == 'val':
            _check_val_fingerprint(fingerprint)
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
