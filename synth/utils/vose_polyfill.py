from typing import Optional, Union
import numpy as np


class PythonSampler:
    """
    Alias method (Vose) for sampling from a discrete distribution.

    The compiled `vose.Sampler` (github.com/Theomat/vose, 0.0.4) is not used:
    its fair die overflows a 32-bit product for n >= 3 and only returns 0 or 1.
    """

    def __init__(self, weights: np.ndarray, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed or 1)
        weights = np.asarray(weights, dtype=float)
        n = len(weights)
        assert n > 0, "cannot sample from an empty distribution"
        total = np.sum(weights)
        assert total > 0, "weights must have a positive sum"
        # Scale so that the average is 1.
        scaled = weights * n / total
        alias = np.zeros(n, dtype=int)
        proba = np.zeros(n, dtype=float)
        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            less = small.pop()
            more = large.pop()
            proba[less] = scaled[less]
            alias[less] = more
            # The column of `less` is filled with `more`.
            scaled[more] = scaled[more] + scaled[less] - 1.0
            if scaled[more] < 1.0:
                small.append(more)
            else:
                large.append(more)
        # Due to floating point inaccuracies, either list may hold the last
        # columns: they are full.
        for i in small + large:
            proba[i] = 1.0
            alias[i] = i
        self.n = n
        self.alias = alias
        self.proba = proba

    def sample_1(self) -> int:
        # Fair die to choose the column, then biased coin within the column.
        col = int(self.rng.integers(0, self.n))
        if self.rng.random() < self.proba[col]:
            return col
        return int(self.alias[col])

    def sample(
        self, k: int = 1, values: Optional[np.ndarray] = None
    ) -> Union[int, np.ndarray]:
        """Sample a random integer or a value from a given array.

        Parameters:
            k: The number of integers to sample. If `k = 1`, then a single int (or float if values is not None) is returned. In any
                other case, a numpy array is returned.
            values: The numpy array of values from which to sample from.

        """
        if values is None:
            if k == 1:
                return self.sample_1()
            return np.asarray([self.sample_1() for _ in range(k)])
        else:
            if k == 1:
                return values[self.sample_1()]  # type: ignore
            return np.asarray([values[self.sample_1()] for _ in range(k)])


Sampler = PythonSampler
