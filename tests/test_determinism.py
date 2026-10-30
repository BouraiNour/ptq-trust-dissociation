import numpy as np
import pytest

def test_bootstrap_seeded_determinism():
    """Ensures the seed fixes the random state so CIs are reproducible."""
    n_samples = 100
    def generate_samples(seed):
        np.random.seed(seed)
        return np.random.choice(n_samples, n_samples, replace=True)
    
    s1 = generate_samples(42)
    s2 = generate_samples(42)
    
    assert np.array_equal(s1, s2), "Bootstrap indices must be deterministic across runs with the same seed."
