"""Tests for src/utils/seeding.py."""
import numpy as np

from src.utils.seeding import seed_everything


def test_seed_deterministic_numpy() -> None:
    """Same seed should produce same numpy random numbers."""
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(42)
    b = np.random.rand(10)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_results() -> None:
    """Different seeds should produce different random numbers."""
    seed_everything(42)
    a = np.random.rand(10)
    seed_everything(123)
    b = np.random.rand(10)
    assert not np.array_equal(a, b)
