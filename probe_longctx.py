"""Fase H -- long-context probe (gates T7.1, T7.2, T7.3).

This is the "drawbacks" section the task asks for literally: the place where the hybrid is
expected to LOSE. Hiding it would make the work worse, so the probe is built to be able to
show a loss, and to say so when it cannot measure anything at all.

The mandatory pre-check comes first (threats.md E8). Before comparing cells, cell 1 -- the
plain global-attention baseline -- must solve the task above chance. If it does not, a flat
curve across all cells says nothing about SWA: it says the model is too small for the task,
and the honest move is to change the task, not the conclusion.

    python probe_longctx.py --check-e8 --ckpt <dir>          # E8 gate, cell 1 only
    python probe_longctx.py --task needle --ckpt-root <dir>  # T7.1 across cells
"""
import argparse
import csv
import json
import os
from contextlib import nullcontext

import torch

import probe_tasks as P
from model import GPT, GPTConfig

CELLS = ['1_mha_full', '2_mha_swa', '3_mla_full', '4_mla_swa',
         '5_gqa_full', '7_mla_all_local']


def load(ckpt_dir, device, impl=None):
    ckpt = torch.load(os.path.join(ckpt_dir, 'ckpt.pt'), map_location=device,
                      weights_only=False)
    args = dict(ckpt['model_args'])
    if impl:
        args['attn_impl'] = impl
    model = GPT(GPTConfig(**args))
    sd = ckpt['model']
    for k in list(sd):                      # torch.compile prefix (sample.py does the same)
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ckpt


@torch.no_grad()
def accuracy(model, x, answers, device, ctx):
    """Two accuracies, because they answer different questions.

    `restricted` argmaxes over the 64 value symbols only: chance is 1/64, and it measures
    retrieval given that the model knows the answer is a value. `open` argmaxes over the
    whole vocabulary: it also requires the model to have learnt the FORMAT. A model that is
    good at the task but never fine-tuned on it scores well on the first and zero on the
    second, and reporting only one of them would misrepresent what happened.
    """
    val_ids = torch.tensor(P.VAL_IDS, device=device)
    x = x.to(device)
    with ctx:
        logits, _ = model(x)
    last = logits[:, -1, :].float()
    pred_open = last.argmax(-1).cpu()
    pred_restricted = val_ids[last[:, val_ids].argmax(-1)].cpu()
    return ((pred_restricted == answers).float().mean().item(),
            (pred_open == answers).float().mean().item())


def sweep(model, task, seq_len, n_depths, n_per_depth, device, ctx, seed=0):
    """Accuracy as a function of how far back the fact sits from the query."""
    rows = []
    for i in range(n_depths):
        g = torch.Generator().manual_seed(seed * 1000 + i)
        if task == 'needle':
            # spread the fact from the very start of the haystack to just before the query
            depth = int(round(i * (seq_len - 8) / max(n_depths - 1, 1)))
            x, a, d = P.batch('needle', n_per_depth, seq_len, g, depth=depth)
        else:
            # capped by the number of distinct key symbols, not by the sequence length:
            # keys are drawn without replacement (see probe_tasks.make_assoc_recall)
            n_pairs = min((seq_len - 8) // 2, P.N_SYMBOLS)
            qi = int(round(i * (n_pairs - 1) / max(n_depths - 1, 1)))
            x, a, d = P.batch('assoc_recall', n_per_depth, seq_len, g,
                              n_pairs=n_pairs, query_index=qi)
        r, o = accuracy(model, x, a, device, ctx)
        rows.append(dict(distance=int(d[0]), restricted=r, open=o))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="a single checkpoint directory (E8 pre-check)")
    ap.add_argument("--ckpt-root", help="directory holding <cell>_s<seed>/ckpt.pt")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--task", default="needle", choices=["needle", "assoc_recall"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--depths", type=int, default=10)
    ap.add_argument("--per-depth", type=int, default=20)
    ap.add_argument("--impl", default="sdpa_mask")
    ap.add_argument("--check-e8", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # nullcontext, NOT torch.enable_grad(): accuracy() is decorated @torch.no_grad(), and
    # enable_grad would switch autograd back ON inside it, so every forward of the sweep
    # would build a graph nobody uses. train.py, bench.py and sample.py all use nullcontext
    # for this same slot.
    ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
           if device == 'cuda' else nullcontext())
    print(f"device={device} task={args.task} seq_len={args.seq_len} "
          f"chance={P.CHANCE:.4f} ({P.N_SYMBOLS} symbols)")

    if args.check_e8:
        model, ckpt = load(args.ckpt, device, args.impl)
        print(f"E8 pre-check on {args.ckpt} (iter {ckpt['iter_num']}, "
              f"best_val {ckpt['best_val_loss']:.4f})")
        rows = sweep(model, args.task, args.seq_len, args.depths, args.per_depth,
                     device, ctx)
        best = max(r['restricted'] for r in rows)
        mean = sum(r['restricted'] for r in rows) / len(rows)
        for r in rows:
            print(f"  distance {r['distance']:6d}  restricted {r['restricted']:.3f}  "
                  f"open {r['open']:.3f}")
        # binomial 99% upper bound under the null, normal approximation
        n = args.depths * args.per_depth
        se = (P.CHANCE * (1 - P.CHANCE) / n) ** 0.5
        thr = P.CHANCE + 2.576 * se
        verdict = "ABOVE CHANCE" if mean > thr else "AT CHANCE"
        print(f"  mean restricted {mean:.4f}  best {best:.4f}  "
              f"chance {P.CHANCE:.4f}  99% threshold {thr:.4f}  -> {verdict}")
        print(f"E8: {'PASS' if verdict == 'ABOVE CHANCE' else 'FAIL'}")
        return

    out_rows = []
    for cell in args.cells.split(","):
        d = os.path.join(args.ckpt_root, f"{cell}_s{args.seed}")
        if not os.path.exists(os.path.join(d, 'ckpt.pt')):
            print(f"{cell:17s} NO CHECKPOINT at {d}")
            continue
        model, ckpt = load(d, device, args.impl)
        rows = sweep(model, args.task, args.seq_len, args.depths, args.per_depth,
                     device, ctx, seed=args.seed)
        for r in rows:
            print(f"{cell:17s} distance {r['distance']:6d}  restricted {r['restricted']:.3f}"
                  f"  open {r['open']:.3f}")
            out_rows.append(dict(task=args.task, cell=cell, seed=args.seed,
                                 seq_len=args.seq_len, **r))
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    if args.out and out_rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"written {args.out}")


if __name__ == '__main__':
    main()
