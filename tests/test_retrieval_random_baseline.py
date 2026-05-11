import math

from app.data_handling.music_eval_retrieval_vs_random import (
    _random_hit_at_k,
    _strip_music_suffix,
)


def test_random_hit_at_k_matches_hypergeometric():
    n_pool, r, k = 10, 3, 5
    k_eff = min(k, n_pool)
    expected = 1.0 - math.comb(n_pool - r, k_eff) / math.comb(n_pool, k_eff)
    assert abs(_random_hit_at_k(n_pool=n_pool, n_pos=r, k=k) - expected) < 1e-12


def test_random_hit_at_k_all_positive():
    assert _random_hit_at_k(n_pool=5, n_pos=5, k=3) == 1.0


def test_random_hit_at_k_none_positive():
    assert _random_hit_at_k(n_pool=5, n_pos=0, k=3) == 0.0


def test_random_hit_at_k_k_larger_than_nonrelevant():
    """If K > N-R, top-K must include at least one relevant item."""
    assert _random_hit_at_k(n_pool=10, n_pos=3, k=9) == 1.0


def test_strip_music_suffix():
    assert _strip_music_suffix("piano music") == "piano"
    assert _strip_music_suffix("Relaxing MUSIC") == "Relaxing"
    assert _strip_music_suffix("epic cinematic grand orchestral music") == "epic cinematic grand orchestral"
    assert _strip_music_suffix("no suffix here") == "no suffix here"
