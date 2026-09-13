"""Gate T1.4 - resolve_pattern (phase 1)."""
import itertools
import sys, os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import resolve_pattern  # noqa: E402


def test_known_expansion():
    # 'LLLG' tiled over 8 layers -> ratio 3:1, last layer global
    assert resolve_pattern("LLLG", 8) == [True, True, True, False] * 2
    assert resolve_pattern("G", 12) == [False] * 12


@pytest.mark.parametrize("n_layer,pattern",
                         itertools.product([6, 8, 12], ["G", "LG", "LLLG", "LLLLLG"]))
def test_shape_and_last_layer_global(n_layer, pattern):
    p = resolve_pattern(pattern, n_layer)
    assert len(p) == n_layer
    assert p[-1] is False, "the last layer must be global"
    assert not all(p), "at least one global layer is required"


@pytest.mark.parametrize("n_layer,pattern",
                         itertools.product([6, 8, 12], ["G", "LG", "LLLG", "LLLLLG"]))
def test_matches_cyclic_tiling_except_last(n_layer, pattern):
    p = resolve_pattern(pattern, n_layer)
    tiled = [c == "L" for c in (pattern * n_layer)[:n_layer]]
    assert p[:-1] == tiled[:-1], "only the last layer may be overridden"


def test_all_local_string_still_keeps_a_global_layer():
    # 'L' means "local everywhere", but force_last_global still carves out the last
    # layer, so the assert `not all(p)` in resolve_pattern can never fire in this
    # branch. That is by design: an all-local model is only reachable explicitly.
    assert resolve_pattern("L", 8) == [True] * 7 + [False]


def test_all_local_available_explicitly():
    # grid cell 7: the other reading of "hybrid", the only one whose KV cache
    # is strictly constant in T. Reachable, but never by accident.
    assert resolve_pattern("L", 8, force_last_global=False) == [True] * 8


def test_rejects_garbage():
    with pytest.raises(AssertionError):
        resolve_pattern("LXG", 8)
    with pytest.raises(AssertionError):
        resolve_pattern("", 8)
