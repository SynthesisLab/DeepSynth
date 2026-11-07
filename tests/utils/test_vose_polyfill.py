import numpy as np

from synth.utils.vose_polyfill import PythonSampler, Sampler


def test_sampler_is_python() -> None:
    assert Sampler is PythonSampler


def test_distribution() -> None:
    rng = np.random.default_rng(0)
    for n in [1, 2, 3, 10, 50]:
        weights = rng.random(n) + 0.01
        probs = weights / weights.sum()
        sampler = PythonSampler(weights.copy(), seed=42)
        samples = sampler.sample(k=20000)
        freq = np.bincount(samples, minlength=n) / len(samples)
        assert np.all(np.abs(freq - probs) < 0.02), (n, freq, probs)


def test_uniform_covers_all() -> None:
    sampler = PythonSampler(np.ones(10) / 10, seed=1)
    assert set(sampler.sample(k=1000)) == set(range(10))


def test_zero_weight_never_sampled() -> None:
    sampler = PythonSampler(np.array([0.5, 0.0, 0.5]), seed=3)
    assert 1 not in set(sampler.sample(k=5000))


def test_seeding() -> None:
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    a = PythonSampler(probs, seed=7)
    b = PythonSampler(probs, seed=7)
    assert all(a.sample() == b.sample() for _ in range(100))


def test_weights_not_modified() -> None:
    w = np.array([0.1, 0.6, 0.3])
    PythonSampler(w, seed=1)
    assert np.allclose(w, [0.1, 0.6, 0.3])
