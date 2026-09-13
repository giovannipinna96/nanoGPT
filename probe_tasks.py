"""Synthetic long-context tasks for Fase H (gates T7.1, T7.2, T7.3).

Why synthetic and not natural-language needle-in-a-haystack: threats.md E8. A 25M model
trained on ~1B tokens almost certainly cannot retrieve a fact stated in English from a
long context. Every cell would sit at chance, the curve would be flat, and the flatness
would be indistinguishable from "the local mask is not doing anything" -- no signal either
way. The mitigation E8 prescribes is a task a small model CAN learn: a dedicated-vocabulary
retrieval task, fine-tuned on briefly and equally for every cell. finetune_probe.py
implements that fine-tuning, but no script of the campaign calls it (audit B-4): every Fase H
number comes from the grid checkpoints evaluated zero-shot.

Contamination (threats.md E9): if the probe format occurred in the training data the test
would measure memorisation instead of retrieval. The two MARKERS are ids 50257/50258, the
padding nanoGPT adds to round the vocabulary to 50304: they cannot occur in tokenised text,
so the exact sequence shape cannot have been seen. Keys and values are ordinary (rare)
vocabulary ids, disjoint from each other and from the filler range -- the guarantee rests on
the markers, not on them.

Two tasks, both returning (tokens, answer_token, distance_from_query_to_the_answer):

  needle          filler ... KEY_i VAL_i ... filler QUERY KEY_i -> VAL_i
                  one fact buried at a controlled DEPTH in a haystack. Depth is what the
                  receptive-field argument is about, so this is the T7.1 task.

  assoc_recall    KEY_a VAL_a KEY_b VAL_b ... QUERY KEY_j -> VAL_j
                  every position carries signal, which separates the cells more sharply
                  because there is no filler to hide behind (T7.2).
"""
import torch

# Reserved slices of the GPT-2 vocabulary. 50257..50303 are the padding ids nanoGPT adds
# to round the vocabulary to 50304: they can never appear in tokenised text, which makes
# them exactly right for the markers (E9).
QUERY_TOKEN = 50257
SEP_TOKEN = 50258

# Keys and values come from the high, rare end of the real vocabulary: distinct from each
# other, and far from the ids that carry the frequent English tokens the filler uses.
N_SYMBOLS = 64
KEY_IDS = list(range(49000, 49000 + N_SYMBOLS))
VAL_IDS = list(range(49500, 49500 + N_SYMBOLS))
FILLER_LO, FILLER_HI = 1000, 40000     # ordinary tokens, no overlap with keys or values

CHANCE = 1.0 / N_SYMBOLS


def _filler(n, g):
    return torch.randint(FILLER_LO, FILLER_HI, (n,), generator=g)


def make_needle(seq_len, depth, g):
    """One haystack with the fact at `depth` tokens from the START.

    Layout, all lengths exact so that `depth` means what it says:

        [ filler x depth ] [ KEY VAL ] [ filler ] [ SEP QUERY KEY ] -> predict VAL

    The distance returned is (seq_len - 1) - (depth + 1) = seq_len - depth - 2, i.e. from
    the query position (the last token) back to the VALUE the model must reproduce. That
    distance, not the absolute position, is what the local receptive field limits
    (threats.md S3). It is the `distance` column of results/T7.*.csv.
    """
    k = int(torch.randint(N_SYMBOLS, (1,), generator=g))
    v = int(torch.randint(N_SYMBOLS, (1,), generator=g))
    tail = torch.tensor([SEP_TOKEN, QUERY_TOKEN, KEY_IDS[k]])
    fact = torch.tensor([KEY_IDS[k], VAL_IDS[v]])
    n_pre = depth
    n_post = seq_len - len(tail) - len(fact) - n_pre
    assert n_post >= 0, f"depth {depth} does not fit in seq_len {seq_len}"
    toks = torch.cat([_filler(n_pre, g), fact, _filler(n_post, g), tail])
    assert len(toks) == seq_len
    return toks, VAL_IDS[v], seq_len - 1 - depth - 1   # query -> VALUE, see the docstring


def make_assoc_recall(seq_len, n_pairs, query_index, g):
    """`n_pairs` key->value pairs SPREAD ACROSS the sequence, then a query for one of them.

    Keys are sampled WITHOUT replacement: with duplicates the task would have more than one
    correct answer and the accuracy would be unreadable. That caps n_pairs at N_SYMBOLS,
    and the cap is the reason the pairs are spread rather than packed. Packed, 64 pairs
    occupy 128 tokens; at seq_len=1024 they would all sit just before the query and the
    task would only ever probe short distances -- which is precisely what T7.2 exists NOT
    to do ("chiedere il valore di una chiave vista molto prima").

    Spread out, every pair is a distractor for every other one, so unlike the needle task
    there is no stretch of ignorable filler around the answer: the model has to hold the
    right association, not merely notice the only structured thing in the window. That is
    what makes T7.2 separate the cells more sharply than T7.1.
    """
    assert n_pairs <= N_SYMBOLS, (
        f"n_pairs={n_pairs} exceeds the {N_SYMBOLS} distinct key symbols; keys are drawn "
        "without replacement so the query would have several correct answers")
    assert 0 <= query_index < n_pairs
    assert 2 * n_pairs + 3 <= seq_len
    keys = torch.randperm(N_SYMBOLS, generator=g)[:n_pairs]
    vals = torch.randint(N_SYMBOLS, (n_pairs,), generator=g)

    toks = _filler(seq_len, g)
    body_room = seq_len - 3                      # everything before [SEP QUERY KEY]
    stride = body_room / n_pairs                 # even spacing across the whole prefix
    pos = [int(i * stride) for i in range(n_pairs)]
    for i, pstart in enumerate(pos):
        toks[pstart] = KEY_IDS[int(keys[i])]
        toks[pstart + 1] = VAL_IDS[int(vals[i])]
    toks[-3:] = torch.tensor([SEP_TOKEN, QUERY_TOKEN, KEY_IDS[int(keys[query_index])]])

    # distance from the query back to the VALUE it must reproduce
    dist = (seq_len - 1) - (pos[query_index] + 1)
    return toks, VAL_IDS[int(vals[query_index])], dist


def batch(task, n, seq_len, g, **kw):
    """Stack `n` independent examples. Returns (x, answers, distances)."""
    fn = {'needle': make_needle, 'assoc_recall': make_assoc_recall}[task]
    xs, ans, dist = [], [], []
    for _ in range(n):
        t, a, d = fn(seq_len, g=g, **kw)
        xs.append(t); ans.append(a); dist.append(d)
    return torch.stack(xs), torch.tensor(ans), torch.tensor(dist)
