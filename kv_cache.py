"""Hybrid KV cache: latent compression (MLA) x windowed allocation (SWA).

This file is the original engineering contribution of the project. None of the three
implementations audited in docs/audit_implementazioni.md has it:

  * nanoGPT has no KV cache at all;
  * nanochat allocates torch.zeros(num_layers, B, seq_len, H, D), i.e. FLAT for every
    layer, so its sliding window saves compute and read bandwidth but NOT a single byte
    of allocated cache;
  * mla-experiments caches the latent but concatenates without bound, so there is no
    rolling buffer.

The combination is what produces the headline number of the report, and it is exactly
what vLLM calls a hybrid KV cache manager: two pools, sized per layer.

    capacity[layer] = window_size   if the layer is LOCAL   -> rolling buffer,
                                                               memory CONSTANT in T
                    = max_seq_len   if the layer is GLOBAL  -> occupied memory grows
                                                               with T, compressed by MLA;
                                                               writing past it RAISES

Threats this file exists to avoid:

  C1  RoPE applied with the rolling-buffer SLOT index instead of the absolute position.
      Symptom: generation is sensible for the first W tokens and then degenerates.
      Mitigation: k^R is rotated by the caller BEFORE being written, at its true
      absolute position, and the cache stores the already-rotated value. The position
      is then frozen inside the cached value and the buffer never needs to know it
      (remediation.md #2 + #10, which fuse into a single fix).
  C2  reading a wrapped buffer in slot order instead of chronological order.
      Mitigation: once a buffer has wrapped, read() gathers with
      (start + arange(n)) % capacity; before that it returns a view, which is already in
      order. Either way it also returns the absolute position of the first key, which
      the mask needs.
  C3  caching reconstructed k_nope/v instead of the latent. That gives perfect quality
      and MHA-sized memory, i.e. it silently cancels the entire point of the project.
      Mitigation: the latent layer type physically has no field to put them in, and
      gate T4.3 asserts it.
  C4  giving local layers max_seq_len capacity. Mitigation: capacity is derived from
      the pattern in one place, and T4.3 asserts it.
  C5  comparing megabytes across different dtypes. Mitigation: n_elements() is reported
      alongside n_bytes().
"""

from dataclasses import dataclass

import torch


@dataclass
class CacheSpec:
    """Everything needed to size a cache. Plain data: the module imports no model code,
    and from_config() below reaches for resolve_pattern only when it is called."""
    n_layer: int
    is_local: list          # per layer, True = sliding window
    window_size: int
    max_seq_len: int
    attn_type: str          # 'mha' | 'gqa' | 'mla'
    # dense layers (mha/gqa)
    n_kv_head: int = 0
    head_dim: int = 0
    v_head_dim: int = 0
    # latent layers (mla)
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    @classmethod
    def from_config(cls, cfg, max_seq_len):
        """The one mapping from a GPTConfig to a cache spec.

        Two fields carry the whole risk of this mapping, and hand-written copies of it had
        already drifted on both (see commit 369ed09, which replaced nine of them):

        * `is_local` must resolve the pattern WITH cfg.force_last_global. Leaving it out
          lets resolve_pattern force the last layer global, which turns cell 7 -- the
          all-local reading, the only configuration whose cache is constant in T -- into a
          variant of cell 4, with no error anywhere (remediation.md #8).
        * `v_head_dim` must come from the config. Today that is belt and braces: MLA does
          not cache v at all, and for mha/gqa GPTConfig pins v_head_dim to head_dim, so a
          hardcoded `head_dim` cannot yet produce a wrong size. It would the moment
          elements_per_token starts reading it for a latent layer.

        resolve_pattern is imported here rather than at module level on purpose: this file
        deliberately does not depend on model.py when it is imported (see CacheSpec above),
        and the pattern is the only thing it needs from it.
        """
        from model import resolve_pattern
        return cls(n_layer=cfg.n_layer,
                   is_local=resolve_pattern(cfg.attn_pattern, cfg.n_layer,
                                            cfg.force_last_global),
                   window_size=cfg.window_size, max_seq_len=max_seq_len,
                   attn_type=cfg.attn_type, n_kv_head=cfg.n_kv_head,
                   head_dim=cfg.n_embd // cfg.n_head, v_head_dim=cfg.v_head_dim,
                   kv_lora_rank=cfg.kv_lora_rank,
                   qk_rope_head_dim=cfg.qk_rope_head_dim)

    def capacity(self, layer):
        return self.window_size if self.is_local[layer] else self.max_seq_len

    def elements_per_token(self, layer):
        """Cache elements a single token occupies in this layer."""
        if self.attn_type == 'mla':
            # one shared latent for k and v, plus the shared RoPE channel (eq. 9/15/17)
            return self.kv_lora_rank + self.qk_rope_head_dim
        # keys and values, one set per kv head
        return self.n_kv_head * (self.head_dim + self.v_head_dim)

    def elements(self, seq_len, batch_size=1):
        """Analytic cache size for a context of seq_len tokens. Exact, kernel-independent."""
        total = 0
        for layer in range(self.n_layer):
            occupied = min(seq_len, self.capacity(layer))
            total += occupied * self.elements_per_token(layer)
        return total * batch_size

    def bytes(self, seq_len, batch_size=1, bytes_per_element=2):
        return self.elements(seq_len, batch_size) * bytes_per_element


class LayerCache:
    """One layer's buffer: one named tensor per cached field.

    Two fields for a dense layer (k, v) and for MLA in its decoupled modes (c_kv, k_rope);
    ONE for rope_mode='reconstructed', where the latent is the whole cache (see
    HybridKVCache.__init__).

    A LOCAL layer's buffer is a ring: it wraps and keeps the last `capacity` = W tokens,
    which is exactly its window. A GLOBAL layer's buffer must never wrap -- wrapping would
    keep only the last `capacity` tokens and silently turn the layer into a sliding window
    -- so with `wraps=False` an overflowing write raises instead (audit M-5).
    """

    def __init__(self, fields, batch_size, capacity, device, dtype, *, wraps):
        # fields: {name: per-token shape}, e.g. {'k': (n_kv_head, head_dim)} or
        # {'c_kv': (kv_lora_rank,)}. One entry or two, see HybridKVCache.__init__.
        self.capacity = capacity
        self.wraps = wraps
        self.buffers = {name: torch.zeros(batch_size, capacity, *shape,
                                          device=device, dtype=dtype)
                        for name, shape in fields.items()}

    def write(self, start_pos, **tensors):
        """Write T new tokens whose FIRST one sits at absolute position start_pos.

        If the chunk is longer than the buffer, only its last `capacity` tokens are
        written. This is not an optimisation: with a longer chunk the wrapped index
        vector contains DUPLICATES, and assignment with duplicate indices has no defined
        winner in PyTorch. In practice it kept the first write, so a prompt longer than
        the window silently left stale keys in the buffer -- correct for the first W
        tokens of the prompt and wrong afterwards. Trimming makes the index vector a
        permutation, which is well defined, and keeps exactly the tokens the window can
        still see (threats.md C2/C7).
        """
        first = next(iter(self.buffers))
        device = self.buffers[first].device
        n = next(iter(tensors.values())).size(1)
        if not self.wraps and start_pos + n > self.capacity:
            raise ValueError(
                f"global-layer cache overflow: positions {start_pos}..{start_pos + n - 1} do "
                f"not fit a buffer of {self.capacity}. Wrapping would silently turn a global "
                "layer into a sliding window; size the cache with a larger max_seq_len.")
        if n > self.capacity:
            skip = n - self.capacity
            start_pos += skip
            tensors = {name: t[:, skip:] for name, t in tensors.items()}
            n = self.capacity
        idx = (start_pos + torch.arange(n, device=device)) % self.capacity
        for name, t in tensors.items():
            self.buffers[name][:, idx] = t.to(self.buffers[name].dtype)

    def read(self, cur_pos):
        """Return the buffers in CHRONOLOGICAL order plus the first absolute position.

        cur_pos is the number of tokens written so far. Until the buffer wraps, slots
        0..cur_pos-1 already hold positions 0..cur_pos-1 in order, so the chronological
        read is a VIEW and copies nothing. That is every global layer, always. Gathering
        there instead copied the whole cache of every layer at every decode step, which
        made the dense-cache cells up to 2.9x slower to decode than they are (audit A-2).
        After a wrap, slot 0 is not token 0, so the gather below is what keeps the mask
        and the ordering honest (threat C2); on a local layer it touches W entries.
        """
        n = min(cur_pos, self.capacity)
        start = cur_pos - n
        if cur_pos <= self.capacity:
            return {name: buf[:, :n] for name, buf in self.buffers.items()}, start
        dev = self.buffers[next(iter(self.buffers))].device
        idx = (start + torch.arange(n, device=dev)) % self.capacity
        return {name: buf[:, idx] for name, buf in self.buffers.items()}, start


class HybridKVCache:
    """Per-layer cache whose capacity AND content both follow the architecture."""

    def __init__(self, spec, batch_size, device, dtype=torch.bfloat16):
        self.spec = spec
        self.batch_size = batch_size
        self.dtype = dtype
        self.pos = 0                       # absolute position of the next token
        self.layers = []
        for layer in range(spec.n_layer):
            if spec.attn_type == 'mla':
                # rope_mode='reconstructed' has no decoupled channel: the latent is the
                # whole cache, and a zero-width k_rope buffer would be a silent lie in
                # every memory number below (it would also be a legal empty tensor, which
                # is exactly how such a lie survives).
                fields = {'c_kv': (spec.kv_lora_rank,)}
                if spec.qk_rope_head_dim > 0:
                    fields['k_rope'] = (spec.qk_rope_head_dim,)
            else:
                fields = {'k': (spec.n_kv_head, spec.head_dim),
                          'v': (spec.n_kv_head, spec.v_head_dim)}
            self.layers.append(LayerCache(fields, batch_size, spec.capacity(layer),
                                          device, dtype, wraps=spec.is_local[layer]))

    # -- bookkeeping ------------------------------------------------------------
    def advance(self, n_tokens):
        self.pos += n_tokens

    def reset(self):
        self.pos = 0

    def n_elements(self):
        """Elements actually allocated (not just occupied)."""
        return sum(b.numel() for layer in self.layers for b in layer.buffers.values())

    def n_bytes(self):
        return sum(b.numel() * b.element_size()
                   for layer in self.layers for b in layer.buffers.values())

    def occupied_elements(self):
        """Elements holding real tokens right now: the analytic quantity of T5.1."""
        return self.spec.elements(self.pos, self.batch_size)

    def capacities(self):
        return [layer.capacity for layer in self.layers]
