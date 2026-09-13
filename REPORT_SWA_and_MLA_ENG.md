# Hybrid SWA + MLA attention on nanoGPT: implementation and controlled comparison

Sliding window attention (SWA) and latent compression (MLA) solve different problems, and their
memory savings multiply. Almost all of the quality cost is paid by the compression, not by the
window, and at this scale the most cost-effective configuration is not the hybrid but
`MHA + SWA`. The hybrid becomes competitive only with absorbed decode, and only when the number of
sequences held in memory is what matters.

## Contents

1. [Objective](#1-objective)
2. [Introduction](#2-introduction)
3. [Related work](#3-related-work)
4. [Methods](#4-methods)
5. [Experimental setup](#5-experimental-setup)
6. [Results](#6-results)
7. [Conclusions](#7-conclusions)
8. [Reproduction](#8-reproduction)
9. [References](#9-references)
10. [MIT License](#10-mit-license)

---

## 1. Objective

> *"For the coding challenge, your task is to implement a hybrid attention mechanism that combines
> both Sliding Window Attention (arxiv.org) and Multi-head Latent Attention (arxiv.org, Section
> 2.1). We recommend beginning with a minimal codebase like nanoGPT (github.com) to get started. We
> also ask to compare this hybrid mechanism against standard attention, outlining the main
> advantages and main drawbacks regarding MLA and SWA."*


## 2. Introduction

Starting from nanoGPT [1], this work implements MLA, SWA, GQA, a **per-layer sized KV cache**
(a ring buffer of $W$ tokens in local layers, a full buffer in global ones) and MLA's **absorbed
decode**, together with automated tests and a measurement harness for quality, memory, speed and
long context.

**What "hybrid" means here.** SWA decides *which* positions a query sees, MLA *how* keys and values
are represented. The hybrid is **across layers**: every layer uses MLA, and layers alternate between
local (a window of $W$ tokens) and global following the pattern `LLLG`, with the last layer global,
as in Gemma 2 and 3 [9, 10]. The alternative reading, **all-local**, with every layer local, was
measured too. To attribute each effect to one mechanism, the comparison is a **2×2 grid** (MHA/MLA ×
global/local), which also makes their interaction measurable.

Besides standard attention (MHA), the comparison includes **GQA**, the de facto standard for reducing
the KV cache and the baseline DeepSeek-V2 presents MLA against.

**18 models with 51 M parameters** were trained on 983 M tokens of FineWeb-Edu [20], on NVIDIA A100
GPUs. All efficiency measurements were taken in the same job for every cell.

---

## 3. Related work

Each entry expands with a click.

<details>
<summary><b>Attention and KV cache</b></summary>

Attention [2] computes $\mathrm{softmax}\left(QK^\top/\sqrt{d} + M\right)V$, with $M$ the causal
mask. Multi-head attention (MHA) uses $n_h$ heads of $d_h$ dimensions. It costs $O(T^2)$ in training
and prefill; in decode it caches keys and values, $2 \cdot n_h \cdot d_h$ elements per token per
layer. With long context and large batches, every decode step is dominated by reading the cache [3].

</details>

<details>
<summary><b>MQA and GQA</b></summary>

MQA [4] shares a single key/value pair across all heads, at some cost in
quality. GQA [5] splits the query heads into $n_{kv}$ groups, each with its own pair: cache
$2 \cdot n_{kv} \cdot d_h$, quality close to MHA with few groups. It is the de facto standard, used
by Llama 2 and Llama 3 [6, 7], Mistral 7B [8], Gemma 2 and 3 [9, 10] and Qwen2 [12]: a few lines of
code, no extra projection, native kernel support [14]. It reduces the cache by giving up heads, i.e.
capacity.

</details>

<details>
<summary><b>Sliding Window Attention</b></summary>

Each query sees only the last $W$ tokens ($0 \le i - j < W$). The idea
of a sparse attention pattern comes from [15, 16]; Mistral 7B [8] uses it with $W = 4096$.

- **Compute:** $O(T \cdot W)$ instead of $O(T^2)$, if the kernel skips the masked blocks.
- **Memory:** the cache of a local layer becomes a *ring buffer* of constant size.
- **Price:** tokens beyond the window arrive only through later layers.

Gemma 2 and 3 alternate local and global layers (1:1 and 5:1) to keep direct access to the whole
context. Here the window counts $W$ positions **including** the current token.

</details>

<details>
<summary><b>Multi-head Latent Attention</b></summary>

DeepSeek-V2 [11], also used in DeepSeek-V3 [13], **compresses** keys and values into a shared latent
$c^{KV} = W^{DKV} h$ of rank $d_c$, from which it reconstructs $k^{C} = W^{UK} c^{KV}$ and
$v = W^{UV} c^{KV}$. Only the latent goes into the cache, with an RMSNorm applied on top.

</details>

<details>
<summary><b>Decoupled RoPE</b></summary>

RoPE [17] rotates pairs of dimensions by an angle proportional to the position, so that
$q_m^\top k_n$ depends only on $m - n$. Rotating $k^{C}$ would put a position-dependent rotation
between the query and $W^{UK}$, and decode would have to reconstruct the whole prefix.
DeepSeek-V2 therefore uses a **separate positional channel** (decoupled RoPE):

- **Two parts:** queries and keys are $[\,\text{content}\ ;\ \text{RoPE}\,]$, with the content part
  never rotated.
- **Shared positional key:** $k^{R}$ (dimension $d^R_h$) is computed from $h$ and shared across
  heads.
- **Softmax scale:** $1/\sqrt{d_h + d^R_h}$.
- **Cache per token per layer:** $d_c + d^R_h$ elements.

</details>

<details>
<summary><b>Absorption (decode on the latent)</b></summary>

Since

$\displaystyle q^{C\top}\left(W^{UK} c\right) = \left(W^{UK\top} q^{C}\right)^{\top} c \qquad \text{and} \qquad \sum_s a_s W^{UV} c_s = W^{UV} \sum_s a_s c_s ,$

the reconstruction can be applied to the **query** and to the **output** instead of to **every
cached token**. Attention runs directly on the latent: same function, same weights, different
order. The naive form pays the projection per cached token ($S$), the absorbed form per query
($T$). The break-even point is at

$\displaystyle T^{\ast} = \frac{d_c\,(d_{\text{nope}} + d_v)}{2d_c + d^R_h - d_{qk} - d_v} = 36.6 \ \text{query}$

in the configuration used. Decode ($T = 1$) wants the absorbed form; prefill and training want the
naive one.

</details>

<details>
<summary><b>Kernels</b></summary>

- **FlashAttention** [14, 18] supports windows and GQA, but requires the same head dimension for
  $q$, $k$ and $v$.
- **FlexAttention** [19] skips the blocks outside the window, but in PyTorch 2.6 it wants
  power-of-two dimensions.
- **FlashMLA:** the production MLA kernels require GPUs newer than the A100 and were not used in
  this report.

</details>

---

## 4. Methods

The repository is a fork of nanoGPT. The training loop, the model skeleton, `configurator.py` and
the data preparation come from nanoGPT; everything else was written for this work.

<div align="center">

| file | role |
|---|---|
| `model.py` | MHA, GQA, MLA, SWA masks, layer patterns, RoPE, attention backends, absorbed decode |
| `kv_cache.py` | per-layer sized KV cache |
| `train.py`, `configurator.py` | new hyperparameters, fixed validation set, training seed |
| `cells.py`, `config/grid_*.py` | the experimental cells |
| `bench_inference.py`, `bench.py` | memory, latency, maximum batch, training throughput |
| `probe_tasks.py`, `probe_longctx.py` | synthetic long-context probes |
| `analysis/*.py`, `scripts/*.py` | statistics, tables and figures; regression oracle against nanoGPT |
| `tests/` | 21 test files |

</div>

<details>
<summary><b>4.1 Attention</b></summary>

**A single entry point.** `attend()` in `model.py` is the only function that computes attention,
for every variant. The local/global choice enters **only** as `is_local`/`window`: MLA changes how
$q$, $k$ and $v$ are produced, SWA which positions are visible.

**Backends.**

- **`flex`** (FlexAttention) for **all** training runs: the kernel is constant across cells, so
  speed differences come from the architecture. MLA's score width (48) is padded to 64 with zeros,
  with the scale passed explicitly ($1/\sqrt{48}$): without it, the padding would silently shift it.
- **`sdpa_mask`**, SDPA with a dense mask: it is the **oracle** of the tests and the backend of
  **all inference measurements**.

**`MultiHeadLatentAttention`** follows the DeepSeek-V2 equations:

- **Cache:** holds only $c^{KV}$ and $k^{R}$.
- **$k^{R}$:** computed from $h$, not from the latent, and shared across heads.
- **Scale:** $1/\sqrt{d_{\text{nope}} + d_{\text{rope}}}$.
- **RMSNorm on the latent:** computed in fp32.
- **Queries:** not compressed, as in DeepSeek-V2-Lite.
- **RoPE channel,** with `rope_mode`:
  - `additive` (default, the paper's): full-width content, $d_{qk} = 48$;
  - `carved`: channel carved out of the head, $d_{qk} = d_v = 32$, no padding;
  - `reconstructed`: no channel.
- **Absorbed decode** (`_forward_absorbed`): reads `kv_up` as $n_h$ separate blocks, without mixing
  heads, and adds no parameters. `absorb_mode='auto'` applies absorption below $T^{\ast}$, i.e. at
  every decode step.

**Details that prevent silent bugs.**

- `is_causal=True` is never used: SDPA would ignore the mask and the window would disappear.
- **RoPE:** the frequencies are computed on the dimension the rotation is applied to, with
  positions always absolute.
- **Init:** the $1/\sqrt{2L}$ scale of the residual projections is applied through an explicit
  marker and not by name, otherwise MLA's output projection would be left out.
- **FLOPs and MFU:** they account for the window and for the different score and value widths.

</details>

<details>
<summary><b>4.2 Per-layer sized KV cache (<code>kv_cache.py</code>)</b></summary>

Each layer has its own capacity:

- **local:** $W$, a ring buffer with constant memory;
- **global:** $T_{\max}$, a buffer that raises an error if exceeded instead of silently turning
  into a window.

**Content:** $k$ and $v$ for MHA/GQA, only $c^{KV}$ and $k^{R}$ for MLA. **Reading:** in
chronological order, with a zero-copy view until the buffer wraps around. **Positions:** keys are
rotated to their absolute position before being written.

</details>

<details>
<summary><b>4.3 Training (<code>train.py</code>)</b></summary>

nanoGPT estimates the validation loss on different random batches at every evaluation. Different
architectures consume the RNG differently at init, so **every cell was validating on different
tokens**, with noise of the same order as the differences being measured (0.01–0.05 nats).
`build_fixed_val` draws 200 batches once, identical for every cell and seed, and prints their
fingerprint, identical across all runs. `train_seed` makes the seed configurable.

</details>

<details>
<summary><b>4.4 Tests</b></summary>

The full suite gives **304 tests passed on A100** and 221 passed plus 44 skipped on CPU. The tests
compare the fast path with the dense oracle or with the mathematical definition.

<div align="center">

| group | files | what they guarantee |
|---|---|---|
| mechanisms | `test_swa`, `test_backends`, `test_pattern`, `test_rope`, `test_mla`, `test_gqa_native`, `test_absorption`, `test_flash` | window of exactly $W$ tokens; FlexAttention ≡ oracle; correct RoPE; MLA shapes, scale and gradients; native GQA ≡ repeated; absorbed decode ≡ naive for 48 steps |
| cache | `test_kv_cache`, `test_cache_parity`, `test_cache_capacity`, `test_cache_zero_copy`, `test_decode_masks` | incremental decode ≡ full forward (error < 1e-4, also after the buffer wraps around); memory ≡ formula; zero-copy |
| measurements | `test_bench_no_grad`, `test_bench_reference`, `test_sdpa_kernel_policy`, `test_cells` | benchmarks without autograd; ratios to MHA; a kernel choice that does not change the results |
| training | `test_fixed_val`, `test_train_checkpoints`, `test_mla_init`, `test_probe_tasks` | identical validation set across cells; hyperparameters in the checkpoints; well-formed synthetic tasks |

</div>

</details>

---

## 5. Experimental setup

**Hardware and software.**

- **GPU:** NVIDIA A100 80 GB and 40 GB, same sm80 architecture. All efficiency measurements on
  80 GB.
- **Software:** Python 3.12, PyTorch 2.6.0+cu124; dependencies pinned in `uv.lock`.

<details>
<summary><b>Model and training parameters (click to expand)</b></summary>

<div align="center">

| group | parameter | value |
|---|---|---|
| model | layers / heads / `n_embd` | 8 / 16 / 512 (`head_dim = 32`), context 1024, GPT-2 vocabulary |
| | positions | RoPE ($\theta = 10\,000$) in every cell |
| SWA | window / pattern | $W = 256$ / `LLLG`, last layer global |
| MLA | latent / channels | $d_c = 256$; $d_{\text{nope}} = 32$, $d_{\text{rope}} = 16$, $d_v = 32$; uncompressed queries |
| GQA | KV heads | 4 (cache of 256 elements against 272 for MLA: equal memory) |
| training | data | FineWeb-Edu `sample-10BT`, 2 shards (1.5 B tokens, median document 629 tokens) |
| | budget | $491\,520$ tokens/iter $\times$ 2000 iter $\approx$ **983 M tokens** |
| | optimizer | AdamW (0.9, 0.95), wd 0.1, lr 6e-4, warmup 200, cosine → 6e-5, bf16, FlexAttention, `torch.compile` |
| | seeds | 1337 and 2024 (plus 3141 for ④ and ⑨) |

</div>

</details>

$d_c = 256$ puts MLA at parameter parity with MHA (+0.27% on the transformer body). GQA-4 instead
has 12.5% fewer parameters: it is matched on memory, not on capacity.

<details>
<summary><b>The experimental cells (click to expand)</b></summary>

<div align="center">

| | cell | attention | pattern | what it isolates |
|---|---|---|---|---|
| ① | `1_mha_full` | MHA | `G` | baseline |
| ② | `2_mha_swa` | MHA | `LLLG` | **SWA** |
| ③ | `3_mla_full` | MLA | `G` | **MLA** |
| ④ | `4_mla_swa` | MLA | `LLLG` | **the hybrid** |
| ⑤ | `5_gqa_full` | GQA-4 | `G` | equal-cache control |
| ⑦ | `7_mla_all_local` | MLA | all `L` | all-local hybrid |
| ⑧ ⑨ | `*_carved*` | MLA, carved channel | `LLLG` | cost of the carved RoPE channel |

</div>

Cell ⑥ (`6_gqa_swa`: GQA with 2 KV heads and pattern `LLLG`) is used only in the cache tests
(`tests/test_cache_parity.py`) and was not trained.

</details>

---

## 6. Results

**The configurations in the tables.**

<div align="center">

| name | cell | what it is |
|---|---|---|
| MHA | ① | standard attention: every layer sees the whole context and caches the keys and values of all 16 heads |
| +SWA, MHA + SWA | ② | MHA with a sliding window: 6 layers out of 8 see only the last 256 tokens, the other 2 the whole context (`LLLG`) |
| MLA | ③ | keys and values compressed into a shared latent of 256 elements; every layer sees the whole context |
| MLA + SWA, hybrid | ④ | MLA in every layer, with the same `LLLG` alternation as ② |
| GQA-4 | ⑤ | the 16 query heads share 4 key/value pairs: a cache 4 times smaller than MHA's |
| all-local | ⑦ | MLA with all 8 layers windowed: no layer sees beyond the last 256 tokens |
| carved | ⑧ ⑨ | like the hybrid, but with the RoPE channel carved out of the head instead of added |

</div>

**Symbols.**

- **$T$:** context length, in tokens. In the memory and speed measurements it is the number of
  tokens already in the cache at measurement time (8k = 8 192, 32k = 32 768).
- **$B$:** batch, i.e. the number of sequences processed in parallel.
- **naive / absorbed:** MLA's two decode forms (§3). The naive form reconstructs keys and values for
  every cached token, the absorbed form works directly on the latent. Cells without MLA have a
  single form.
- **Numbers in parentheses:** ratio to MHA under the same conditions.

### 6.1 Quality

**What is measured.** The validation loss is the next-token cross-entropy,

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} -\ln p\left(\text{correct token}_i\right),
$$

computed on the same validation set for every cell (§4.3). It is measured in **nats** because it
uses the natural logarithm; $1\ \text{nat} = 1/\ln 2 \approx 1.44$ bits. For reference, MHA's loss
(3.76 nats) corresponds to a perplexity of $e^{3.76} \approx 43$: the model is as uncertain as if it
were choosing at random among about 43 tokens. Choosing at random among all the tokens of the
vocabulary would give $\ln 50\,257 \approx 10.8$ nats.

**How to read the table.** **The lower the loss, the better.** $\Delta$ vs MHA is the difference
from the baseline: negative = better than MHA, positive = worse. Since $e^{\Delta} \approx 1 + \Delta$
for small $\Delta$, $\Delta$ nats correspond to about $\Delta \times 100\%$ of perplexity
($+0.035 \approx +3.6\%$). The seed-to-seed $\sigma$, pooled over the six cells, is **0.0069 nats**:
the significance threshold $2\sigma$ is **0.0138**, and below it a difference cannot be told apart
from seed noise and is not claimed. Results from `results/grid_summary.csv`, 12 runs out of 12, no
NaN:

<div align="center">

| cell | val loss | $\Delta$ vs MHA |
|---|---|---|
| ① MHA | 3.7579 | — |
| ② **MHA + SWA** | **3.7508** | **−0.0070** |
| ③ MLA | 3.8022 | +0.0444 |
| ④ MLA + SWA (hybrid) | 3.7932 | +0.0354 |
| ⑤ GQA-4 | 3.7840 | +0.0261 |
| ⑦ MLA all-local | 3.8063 | +0.0484 |

</div>

- **The window costs nothing measurable; the hybrid's cost comes from MLA, not from the
  mask.** For MLA the learning rate was not retuned and the init scale is not matched with MHA, so
  +0.044 should be read as an upper bound.
- **Interaction** $\Delta_{\text{hybrid}} - (\Delta_{\text{SWA}} + \Delta_{\text{MLA}})$: −0.0192 and
  +0.0153 on the two seeds, mean **−0.0020**. It is not detected, but two seeds would resolve it only
  above ~0.027 nats.
- **At equal cache, GQA-4 beats MLA by 0.0182 nats**, above threshold, with less cache and fewer
  parameters.
- **Carved RoPE channel:** no cost detected. Compared with ④, cell ⑨ has the same positional
  channel and the same cache, and half the content: +0.0012 on two seeds, +0.0044 on three. It
  removes the padding and the kernels that refuse the shapes, and it is the natural candidate as
  the default implementation.

**Why the hybrid costs +0.035.** SWA and MLA act on different things. The window takes distant
tokens away from local layers, but with a context of 1 024 and $W = 256$ most of the information
useful for predicting the next token is nearby, and one global layer out of four recovers the rest:
that is why ② costs nothing. MLA instead compresses the keys and values of all 16 heads into a
single latent of 256 elements, and this limits what each head can represent in **every** layer. The
hybrid therefore inherits almost only MLA's cost: +0.035 against +0.044 for MLA alone, a difference
(0.009) below threshold.

![loss curves](assets/T6.4_loss_curves.png)

**What the figure shows.** Validation loss curves during training, two seeds per cell, unsmoothed.
In both panels the y axis is the validation loss on the fixed validation set (§4.3), limited to
3.7–4.6 to make the final part of training readable.

- **Left panel, per token:** the x axis shows the tokens seen during training, in millions (up to
  983 M). All six cells are shown.
- **Right panel, per second:** the x axis shows the wall-clock training time on one A100, in
  minutes, from the throughput measured in T5.4. Only the four cells whose time was measured are
  shown: MHA, MHA + SWA, MLA and MLA + SWA.

**What to look for.** All curves go down and flatten, and in the final part they stay almost
parallel: the gap between cells is not closing, so the ranking in the table does not depend on
where training stops. On the left, MHA + SWA and MHA end lowest (≈ 3.75), the cells with MLA and
GQA-4 higher, MLA all-local last. The right panel checks that the comparison also holds at equal
time, because a per-token axis would credit a slow cell with an efficiency it does not have: MLA
takes a few minutes longer (≈ 64 against ≈ 61–62 minutes) and stays higher, so at equal time its
disadvantage grows, while MHA + SWA finishes first and lowest.

### 6.2 Memory

**How to read the table.** Each value is the cell's KV cache as a percentage of MHA's at the same
$T$: **lower is better** (25% = a cache 4 times smaller). Being ratios, the percentages depend
neither on the batch nor on the data type. (`results/T5.1_kv_cache_analytic.csv`):

<div align="center">

| $T$ | MHA | +SWA | GQA-4 | MLA | **MLA+SWA** | all-local |
|---|---|---|---|---|---|---|
| 1 024 | 100% | 43.8% | 25.0% | 26.6% | **11.6%** | 6.6% |
| 16 384 | 100% | 26.2% | 25.0% | 26.6% | **7.0%** | 0.4% |
| 131 072 | 100% | 25.1% | 25.0% | 26.6% | **6.7%** | 0.1% |

</div>

- **The savings multiply exactly:** $26.6\% \times 26.2\% = 7.0\%$ at 16k tokens.
- **MLA** gives a constant factor.
- **SWA** saturates at 25%, because with long context the 2 global layers dominate.
- **Without global layers** (⑦) the cache stays at 1.06 MiB per sequence at every length: **a
  single global layer is enough to make the cache $O(T)$**.

**Why the hybrid gets to 7%.** The cache is

$$
\text{cache} = \sum_{\text{layer}} \min\left(T,\ \text{layer capacity}\right) \times \text{elements per token}.
$$

MHA holds $16 \times (32 + 32) = 1024$ elements per token per layer, MLA $256 + 16 = 272$ (26.6%).
With `LLLG` the 6 local layers hold at most 256 tokens and the 2 global ones all $T$: at 16 384
tokens the fraction is

$$
\frac{2 \cdot 16\,384 + 6 \cdot 256}{8 \cdot 16\,384} = 26.2\% .
$$

MLA reduces the elements per token, SWA the number of tokens in the local layers: they act on
different factors of the formula and therefore multiply. The limit is the global layers: with long
context their cache grows with $T$, and the hybrid does not go below $26.6\% \times 25\% \approx 6.7\%$.

### 6.3 Decode speed

Decode throughput with batch 64 and naive decode (`results/v2/T5.2_latency_never.csv`).

**How to read the table.** The number is the throughput, i.e. the tokens generated per second summed
over the 64 sequences, $\text{tok/s} = B \times \text{steps} / \text{time}$: **higher is better**. In
parentheses, the ratio

$$
(\times\,\text{MHA}) = \frac{\text{tok/s of the cell}}{\text{tok/s of MHA}} \quad \text{with the same } T \text{ and } B .
$$

Above 1 the cell is faster than MHA, below 1 slower (0.28 = about $1/0.28 \approx 3.6$ times
slower). Each row has its own denominator, so ratios are compared along a row; across rows, compare
the tok/s.

<div align="center">

| $T$ | MHA | +SWA | GQA-4 | MLA | MLA+SWA | all-local |
|---|---|---|---|---|---|---|
| 1 024 | 11 496 (1.00) | 10 726 (0.93) | 10 939 (0.95) | 7 026 (0.61) | 7 677 (0.67) | 8 125 (0.71) |
| 8 192 | 4 941 (1.00) | 8 589 (1.74) | 9 805 (1.98) | 1 225 (0.25) | 3 688 (0.75) | 7 735 (1.57) |
| 32 768 | 1 289 (1.00) | **4 095 (3.18)** | **4 014 (3.11)** | 355 (0.28) | 1 303 (1.01) | **7 757 (6.02)** |

</div>

- **At batch 1 nothing is faster than MHA** (0.94–0.96× for SWA and GQA, 0.63–0.76× for the MLA
  cells): the step is dominated by the kernels, not by reading the cache.
- **SWA and GQA are equivalent speed levers** with long context and large batches.
- **Naive MLA is a cost everywhere**, because at every step it reconstructs keys and values for the
  whole context.

**Why the hybrid behaves like this.** With long context and large batches every step is dominated by
reading the cache [3]: +SWA and GQA-4 are fast because they read less (at 32k about 4 times less).
Naive MLA instead reconstructs full-width keys and values for **all** cached tokens at every step, a
matrix multiplication per token, which costs more than the reading itself (0.28 at 32k). In the
hybrid the reconstruction weighs fully only in the 2 global layers, while in the 6 local ones it
covers at most 256 tokens: the cost drops by about 4 times and the hybrid gets back to parity with
MHA (1.01), without gaining. At 1 024 tokens the cache is short and the fixed cost of MLA's extra
projections dominates: all MLA cells sit between 0.6 and 0.7.

**Absorbed decode** (`--absorb auto`: absorbed decode, naive prefill;
`results/v2/T5.2_latency_auto.csv`). The table compares the same cell in the two decode forms, so it
covers only the cells with MLA. $B$ is the batch, 64 sequences in parallel in every row; $T$ is the
context.

**How to read the table.** The numbers are tok/s: **higher is better**. In parentheses, the same
$(\times\,\text{MHA})$ ratio as above, with the same $T$ and $B$. MHA has no absorbed form, so in
both columns the denominator is plain MHA, measured in the same benchmark job as the column:
4 941 and 1 289 tok/s (8k and 32k) for naive, 4 939 and 1 294 for absorbed, a difference below
0.4%. For example $2915 / 1294 = 2.25$. Comparing the two columns tells how much absorption gains;
the parenthesis tells where the cell stands relative to MHA.

<div align="center">

| cell | $T$, $B$ | naive (× MHA) | absorbed (× MHA) |
|---|---|---|---|
| MLA | 32k, 64 | 355 (0.28) | 942 (0.73) |
| MLA + SWA | 8k, 64 | 3 688 (0.75) | 5 891 (1.19) |
| MLA + SWA | 32k, 64 | 1 303 (1.01) | **2 915 (2.25)** |
| MLA all-local | 32k, 64 | 7 757 (6.02) | 6 909 (5.34) |

</div>

- **Absorption multiplies MLA's throughput by 2.7–2.8** and brings the hybrid to 2.25× MHA, but MLA
  alone stays below MHA and well below GQA-4.
- **Where the cache does not dominate, absorption costs:** at batch 1 (0.48× for MLA at 32k) and
  with short caches.
- **The absorbed form is only for decode:** applied to a 32k-token prefill with batch 64 it runs out
  of the 80 GB of memory.

**Why the hybrid improves like this.** Absorption moves the projection from every cached token to
the query and the output only (§3): the cost that grew with $T$ in the naive form disappears, and
what remains is reading the latent, 272 elements per token instead of 1 024. The longer the context,
the more the saving counts: the hybrid goes from 1.19 at 8k to 2.25 at 32k. MLA alone stays below MHA
(0.73) because it still reads all $T$ tokens in all 8 layers, with scores computed on wider vectors
(272 elements per head instead of 32). In the all-local cell the cache holds only 256 tokens: there
is little to save and the fixed cost of the absorbed form prevails (6.02 → 5.34).

### 6.4 Resident batch

Maximum number of 8 192-token sequences that fit on an 80 GB A100 during decode. The cache is
brought to 8 192 tokens without a prefill, so the measured limit is that of decode and not that of
the prompt (`results/v2/T5.3b_max_batch_decode_{never,auto}.csv`).

**How to read the table.** The number is the maximum batch: **higher is better**, because it tells
how many sequences (for example users) can be served together on the same GPU. In parentheses, the
ratio to MHA (627 sequences). The absorbed column is empty (—) for MHA, MHA + SWA and GQA-4 because
absorption exists only for MLA: it relies on the latent projections $W^{UK}$ and $W^{UV}$, which the
dense cells do not have.

<div align="center">

| cell | naive | absorbed |
|---|---|---|
| MHA | 627 (1.00×) | — |
| MHA + SWA | 2 265 (3.61×) | — |
| GQA-4 | 2 506 (4.00×) | — |
| MLA | 1 297 (2.07×) | 2 100 (3.35×) |
| MLA + SWA | 2 156 (3.44×) | **5 919 (9.44×)** |
| MLA all-local | 38 672 (61.7×) | 58 503 (93.3×) |

</div>

- **The dense cells are limited by the cache:** the cache takes 77–78 GB of the 80 available.
- **Naive MLA is not:** the full-rank reconstruction of a global layer fills the GPU before the
  cache does.
- **With absorption the hybrid reaches 9.44×**, but MLA alone still holds fewer sequences than
  GQA-4.

**Why the naive hybrid holds fewer sequences than MHA + SWA, and why absorption raises it.** A
back-of-the-envelope estimate in bf16 (2 bytes per element), assuming for every cell the ~78 GiB
budget the cache takes in the dense cells:

<div align="center">

| cell | cache per sequence at 8k | sequences if only the cache counted | measured |
|---|---|---|---|
| MHA | $8 \times 8192 \times 1024 \times 2$ bytes = 128 MiB | ~620 | 627 |
| MHA + SWA | $(2 \cdot 8192 + 6 \cdot 256) \times 1024 \times 2$ bytes = 35 MiB | ~2 280 | 2 265 |
| MLA + SWA | $(2 \cdot 8192 + 6 \cdot 256) \times 272 \times 2$ bytes = 9.3 MiB | ~8 600 | 2 156 naive, 5 919 absorbed |

</div>

For the dense cells estimate and measurement agree: the limit is the cache. The naive hybrid instead
stops at a quarter of the estimate. At every step naive MLA reconstructs full-width keys and values
for all the tokens of a global layer: $8192 \times 16 \times (48 + 32)$ elements (tokens × heads ×
key and value dimension) $\approx 20$ MiB of temporaries per sequence, more than the cache itself.
Divided over the measured sequences, $78\ \text{GiB} / 2156 \approx 37$ MiB per sequence: 9.3 MiB of
cache and the rest for the reconstruction. MLA alone shows the same overhead (~28 MiB). With
absorption the reconstruction disappears, the limit becomes the cache again and the hybrid rises to
5 919 sequences ($78\ \text{GiB} / 5919 \approx 13.5$ MiB each). It stays below the ~8 600 estimate
because of the activations and buffers that do not depend on the cache.


### 6.5 Long context: where the hybrid loses

**The task.** In the needle task the sequence contains a `KEY VALUE` pair and ends with the question
`QUERY KEY`: the model must answer with the value. The distance is the number of tokens between the
pair and the question. Accuracy is the fraction of correct answers among 64 possible values, so the
chance level is $p = 1/64 \approx 0.016$. The baseline solves the task: on the needle its mean
accuracy is 0.305 and 0.220 on the two seeds.

**How to read the first table.** There are 10 sampled distances, from 6 to 1 022 tokens, at steps of
about 113. Using the accuracy averaged over the two seeds, each cell reports the last distance above
the 99% chance threshold and the next sampled distance, where retrieval is already at chance level:
the drop lies between the two. The threshold, with $n = 40$ examples per point, is

$$
p + 2.576 \sqrt{\frac{p\,(1 - p)}{n}} = 0.066 .
$$

For example, 570–683 means that MHA still retrieves at 570 tokens and no longer at 683. **Farther is
better** (`results/T7.1_needle_s*.csv`):

<div align="center">

| MLA | MHA | MHA+SWA | GQA-4 | all-local | **MLA+SWA** |
|---|---|---|---|---|---|
| **909–1022** | 570–683 | 458–570 | 458–570 | 458–570 | **232–345** |

</div>

- **The window is active:** ② and ④ decay before ①.
- **No sharp drop at $3 \cdot W = 768$**, and the all-local cell decays well before its theoretical
  receptive field ($8 \cdot W = 2048$). At this scale the limit is the learned retrieval capacity.
- **MLA has the best long-range retrieval despite having the worst loss.** The hypothesis that this
  depends on the unrotated content channel, tested on the carved cells, is not supported.

**On retrieval the two mechanisms do not add up.** Here the values are **mean accuracies** over all
distances, so **higher is better** (the opposite of the loss). Writing $A_i$ for the mean accuracy of
cell $i$ (① MHA, ② MHA + SWA, ③ MLA, ④ hybrid):

$$
\Delta_{\text{SWA}} = A_2 - A_1 , \qquad
\Delta_{\text{MLA}} = A_3 - A_1 , \qquad
\text{additive prediction} = A_1 + \Delta_{\text{SWA}} + \Delta_{\text{MLA}} ,
$$

$$
\text{interaction} = A_4 - \text{additive prediction} .
$$

The additive prediction is what the hybrid would score if the effects added up. An interaction
close to zero means the effects add up, a negative one that the hybrid does worse than the sum. In
parentheses, the value for each seed.

<div align="center">

| task | $\Delta_{\text{SWA}}$ | $\Delta_{\text{MLA}}$ | additive prediction | measured hybrid | interaction (per seed) |
|---|---|---|---|---|---|
| needle | −0.045 | +0.128 | 0.345 | **0.202** | **−0.143** (+0.005 / −0.290) |
| associative recall | −0.058 | +0.140 | 0.355 | **0.197** | **−0.158** (−0.045 / −0.270) |

</div>

The interaction is zero or negative in all four cases, but the magnitudes vary a lot across seeds,
so it is **a direction, not a measurement**: **the window cancels the long-range advantage MLA has
on its own**. On the other hand, the cells with a window degrade 2–4 times less when the context
doubles, i.e. on the needle at 2 048 tokens, twice the training context (mean accuracy
−0.06/−0.11 against −0.19/−0.24).

**Why the hybrid is the worst at retrieval.** To retrieve a distant value a layer must be able to
read it directly. In ② the window leaves this job to the 2 global layers alone, and ② loses little.
MLA alone has the best retrieval of all cells, although the reason is not clear. In the hybrid the
two combine badly: distant retrieval rests entirely on 2 layers, and in those layers too the keys
and values go through the compressed latent. One possible explanation, **not verified**, is that
MLA's advantage requires spreading retrieval over many layers, which the window prevents. With two
seeds giving +0.005 and −0.290 the direction is consistent, the magnitude is not.

---

## 7. Conclusions

### 7.1 Advantages and drawbacks

**SWA**

<div align="center">

| advantages | drawbacks |
|---|---|
| zero parameters; unchanged quality (−0.007 nats) | worsens long-distance retrieval, and with MLA cancels its advantage |
| constant cache on local layers; batch 3.61× | no speed-up at small batch or short context |
| decode 3.18× at 32k, training +21% at 4k, on standard kernels | memory saving saturates (~25%) as long as global layers remain |
| degrades less as the context grows | training needs a kernel that skips masked blocks |

</div>

**MLA**

<div align="center">

| advantages | drawbacks |
|---|---|
| cache reduced by a constant and exact factor, without approximating attention | +0.044 nats at this scale |
| parameter parity with MHA | slower than MHA in decode at every measured point |
| with absorbed decode the saving turns into batch (9.44× with SWA) | without absorption a 3.8× smaller cache gives only 2.07× batch |
| best measured long-range retrieval | at equal cache **dominated by GQA-4** on loss, batch and throughput |
| | high complexity: more projections, RoPE channel, two decode forms, less compatible kernels |

</div>

**The hybrid.** The cache savings multiply (14.4× at 16k), on loss the costs do not interact
detectably, on retrieval they do worse than adding up. The decisive cost is not per layer: it is
**the existence of any global layer at all**. The two global layers of `LLLG` make the difference
between 5 919 and 58 503 sequences in memory.

### 7.2 Which one to choose

Criteria fixed before the measurements: $\Delta$ loss $< +0.05$, cache at 16k $> 20\times$,
batch $> 5\times$, throughput at 32k $> 2\times$ (`results/v2/T8.1_decision_table*.csv`).

<div align="center">

| cell | $\Delta$ loss | cache 16k | batch 8k (absorbed) | tok/s 32k (absorbed) | criteria met |
|---|---|---|---|---|---|
| **MHA+SWA** | −0.0070 | 3.8× | 3.61× | 3.17× | 2 of 4 |
| GQA-4 | +0.0261 | 4.0× | 4.00× | 3.11× | 2 of 4 |
| MLA | +0.0444 | 3.8× | 3.35× | 0.73× | 1 of 4 |
| MLA+SWA | +0.0354 | 14.4× | 9.44× | 2.25× | 3 of 4 |
| **MLA all-local** | +0.0484 | 241× | 93.3× | 5.34× | 4 of 4 |

</div>

<div align="center">

| situation | choice |
|---|---|
| **default: quality first, unbounded context** | **MHA + SWA**: baseline quality, 3.8× less cache, 3.6× the batch, 3.2× the throughput with long context, zero parameters, standard kernels. It is the only Pareto improvement over MHA |
| same serving numbers without a window | GQA-4, at +0.026 nats |
| maximum number of sequences, unbounded context | **MLA + SWA with absorbed decode**: 2.6× the sequences of MHA+SWA at +0.042 nats, but worse retrieval; without absorption it does not pay off |
| retrieval needed only within ~500 tokens | MLA all-local: 241× less cache, 93× the batch, +0.048 nats |
| MLA alone | not recommended at this scale |

</div>

### 7.3 What to expect with larger models

This section is an **extrapolation**: no model beyond 51 M parameters was trained in this work. The
memory figures are exact arithmetic on hypothetical configurations; the expectations on quality rely
on what published models report, not on our own measurements.

**What changes with scale.**

- **The cache becomes the main constraint.** With 128k tokens of context, the cache of a single
  sequence is of the same order as the model weights, and it decides how many sequences can be
  served per GPU.
- **The baseline to beat is GQA, not MHA.** Almost all large models use GQA (§3).
- **MLA compresses more.** In DeepSeek-V2 and V3 [11, 13] the latent stays small ($d_c = 512$,
  $d^R_h = 64$) while heads and dimensions grow: $d_c + d^R_h = 576$ elements per token per layer,
  against $2 \cdot 8 \cdot 128 = 2048$ for a GQA with 8 groups of 128 dimensions, i.e.
  $2048 / 576 \approx 3.6$ times less. In this work, instead, MLA was matched to the cache of GQA-4,
  and GQA-4 won.
- **The window covers a smaller fraction of the context.** $W = 256$ over 1 024 tokens is a quarter
  of the context; $W = 4096$ over 128k is 3%. The global layers weigh more, both on memory and on
  retrieval.

**An order of magnitude.** Cache for **one** sequence of $T = 131\,072$ tokens in bf16, with $L$
layers, $n_h$ heads of $d_h = 128$ dimensions, and for MLA $d_c = 512$ and $d^R_h = 64$:

$$
\text{MHA} = 2\,L\,n_h\,d_h \cdot T \cdot 2\ \text{bytes}, \qquad
\text{GQA-8} = 2\,L \cdot 8\,d_h \cdot T \cdot 2\ \text{bytes}, \qquad
\text{MLA} = L\,(d_c + d^R_h) \cdot T \cdot 2\ \text{bytes}.
$$

For the hybrid we assume one global layer every 6, as in Gemma 3 [10], and $W = 4096$, as in
Mistral 7B [8]:

$$
\text{MLA + SWA} \approx \text{MLA} \times \left(\frac{1}{6} + \frac{5}{6} \cdot \frac{4096}{131\,072}\right) \approx \text{MLA} \times 0.19 .
$$

<div align="center">

| model (hypothetical configuration) | MHA | GQA-8 | MLA | MLA + SWA |
|---|---|---|---|---|
| 10B dense: 40 layers, 40 heads | 100 GiB | 20 GiB | 5.6 GiB | ~1.1 GiB |
| 100B dense: 80 layers, 64 heads | 320 GiB | 40 GiB | 11.3 GiB | ~2.2 GiB |
| 1T MoE: 61 layers, 128 heads (attention as in DeepSeek-V3) | 488 GiB | 30.5 GiB | 8.6 GiB | ~1.7 GiB |

</div>

The hybrid takes about 18 times less than GQA-8. In an MoE the cache depends only on the attention
and not on the number of experts: that is why the 1T model does not have the largest cache.

**In short.** With scale the hybrid's memory advantages grow, because the cache weighs more and MLA
compresses more relative to GQA, and MLA's quality cost may shrink. The open question is retrieval
with long context: the window covers an ever smaller part of the context, and §6.5 suggests that it
is precisely on retrieval that SWA and MLA do not add up. Before adopting the hybrid in a large
model, the decisive test is retrieval at 128k against GQA + SWA at equal memory.

---

## 8. Reproduction

```bash
source env.sh && uv sync                   # cluster paths in env.sh; dependencies from uv.lock
bash run_all.sh tests                      # test suite
bash run_all.sh cell 4_mla_swa 1337        # one grid run (~1 h on an A100)
bash run_all.sh grid                       # 6 cells × 2 seeds
bash run_all.sh remeasure                  # efficiency -> results/v2/
bash run_all.sh faseh                      # long context
bash run_all.sh analysis                   # tables and figures (CPU only)
```

`bash run_all.sh` without arguments lists every target. **What was verified:**

- **Analysis:** regenerating from the checkpoints, 16 files out of 16 are byte-for-byte identical to
  those in `results/`.
- **Tests:** 304 passed on A100.
- **Regression:** with the default configuration the model reproduces nanoGPT (loss difference
  7e−06 over 20 steps).
- **Benchmarks:** re-run, identical maximum batches and timings within 1.6%.
- **Training:** not bit-for-bit reproducible, because of FlexAttention's non-deterministic
  backward; a new run falls within the seed-to-seed variability.

---

## 9. References

1. A. Karpathy. *nanoGPT*. https://github.com/karpathy/nanoGPT
2. A. Vaswani et al. *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762
3. R. Pope et al. *Efficiently Scaling Transformer Inference*. MLSys 2023. arXiv:2211.05102
4. N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019. arXiv:1911.02150
5. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023. arXiv:2305.13245
6. H. Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. 2023. arXiv:2307.09288
7. A. Grattafiori et al. (Llama Team, AI @ Meta). *The Llama 3 Herd of Models*. 2024. arXiv:2407.21783
8. A. Q. Jiang et al. *Mistral 7B*. 2023. arXiv:2310.06825
9. Gemma Team. *Gemma 2: Improving Open Language Models at a Practical Size*. 2024. arXiv:2408.00118
10. Gemma Team. *Gemma 3 Technical Report*. 2025. arXiv:2503.19786
11. DeepSeek-AI. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*. 2024. arXiv:2405.04434
12. A. Yang et al. *Qwen2 Technical Report*. 2024. arXiv:2407.10671
13. DeepSeek-AI. *DeepSeek-V3 Technical Report*. 2024. arXiv:2412.19437
14. T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. 2023. arXiv:2307.08691
15. R. Child et al. *Generating Long Sequences with Sparse Transformers*. 2019. arXiv:1904.10509
16. I. Beltagy, M. E. Peters, A. Cohan. *Longformer: The Long-Document Transformer*. 2020. arXiv:2004.05150
17. J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. arXiv:2104.09864
18. T. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. arXiv:2205.14135
19. PyTorch Team. *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention*. 2024. https://pytorch.org/blog/flexattention/
20. G. Penedo et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*. 2024. arXiv:2406.17557

---

## 10. MIT License

The repository is distributed under the MIT license. It is a fork of nanoGPT [1], released under the
same license: the copyright notice and the license text, reproduced below and in the `LICENSE` file,
must be kept in every copy or substantial portion of the software.

```text
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
