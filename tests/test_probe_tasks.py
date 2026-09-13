"""Unit tests for the Phase H task generators (probe_tasks).

These exist because the first cluster run of T7.2 died on a generator bug that no local
test could have caught: nothing checked that the requested number of pairs fitted in the
symbol alphabet, and the failure only appeared once a real sweep asked for 508 pairs out
of 64 distinct keys. A probe that silently builds a malformed example is worse than one
that crashes -- it would have produced a plausible accuracy curve for the wrong task.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import probe_tasks as P  # noqa: E402


def _g(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("seq_len", [128, 1024, 2048])
@pytest.mark.parametrize("depth_frac", [0.0, 0.5, 0.9])
def test_needle_answer_follows_the_queried_key(seq_len, depth_frac):
    depth = int(depth_frac * (seq_len - 8))
    x, ans, dist = P.batch('needle', 4, seq_len, _g(), depth=depth)
    assert x.shape == (4, seq_len)
    for row, a in zip(x, ans):
        assert row[-3].item() == P.SEP_TOKEN and row[-2].item() == P.QUERY_TOKEN
        key = row[-1].item()
        assert row[depth].item() == key, "the fact must sit exactly at `depth`"
        assert row[depth + 1].item() == a.item(), "the answer must follow its key"
        assert a.item() in P.VAL_IDS


@pytest.mark.parametrize("seq_len,n_pairs", [(1024, 64), (2048, 64), (256, 32)])
def test_assoc_recall_keys_are_unique_and_answer_is_unambiguous(seq_len, n_pairs):
    for qi in (0, n_pairs // 2, n_pairs - 1):
        x, ans, dist = P.batch('assoc_recall', 2, seq_len, _g(qi),
                               n_pairs=n_pairs, query_index=qi)
        for row, a in zip(x, ans):
            key = row[-1].item()
            hits = [i for i in range(seq_len - 3) if row[i].item() == key]
            assert len(hits) == 1, f"queried key occurs {len(hits)} times, answer ambiguous"
            assert row[hits[0] + 1].item() == a.item()
            keys = [row[i].item() for i in range(seq_len - 3)
                    if row[i].item() in P.KEY_IDS]
            assert len(keys) == len(set(keys)) == n_pairs, "keys must be distinct"


def test_assoc_recall_refuses_more_pairs_than_symbols():
    """The bug that broke job 97832: (seq_len-8)//2 pairs asked of a 64-symbol alphabet."""
    with pytest.raises(AssertionError, match="distinct key symbols"):
        P.batch('assoc_recall', 1, 1024, _g(), n_pairs=508, query_index=0)


def test_distance_is_measured_from_the_query_backwards():
    """Distance is the span the model must bridge, which is what the receptive-field
    argument is stated in -- not the absolute position of the fact."""
    _, _, d_far = P.batch('needle', 1, 1024, _g(), depth=0)
    _, _, d_near = P.batch('needle', 1, 1024, _g(), depth=1000)
    assert d_far[0] > d_near[0]
    _, _, a_far = P.batch('assoc_recall', 1, 1024, _g(), n_pairs=64, query_index=0)
    _, _, a_near = P.batch('assoc_recall', 1, 1024, _g(), n_pairs=64, query_index=63)
    assert a_far[0] > a_near[0]


def test_markers_cannot_collide_with_real_text():
    """Ids 50257/50258 are the padding nanoGPT adds to round the vocab to 50304, so they
    can never occur in tokenised text -- that is what makes the probe contamination-proof.
    Keys and values must also be disjoint from each other."""
    assert P.QUERY_TOKEN >= 50257 and P.SEP_TOKEN >= 50257
    assert not set(P.KEY_IDS) & set(P.VAL_IDS)
    assert max(P.FILLER_LO, 0) > 0 and P.FILLER_HI <= min(P.KEY_IDS)
