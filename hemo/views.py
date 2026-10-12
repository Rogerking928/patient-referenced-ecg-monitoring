"""Lazy transformed views of the waveform array, used by the ablation controls."""
from __future__ import annotations

import numpy as np


class _BaseView:
    def __init__(self, X):
        self.X = X
        self.shape = X.shape
        self.dtype = np.float32

    def __len__(self):
        return len(self.X)

    def _one(self, k):
        raise NotImplementedError

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            return self._one(int(k))
        idx = np.atleast_1d(np.asarray(k))
        return np.stack([self._one(int(i)) for i in idx])


class ShuffledView(_BaseView):
    """Sample order permuted within each window, deterministically per window.

    Amplitude distribution is preserved exactly; every temporal structure --
    rate, variability, morphology -- is destroyed.
    """

    def __init__(self, X, seed=0):
        super().__init__(X)
        self.seed = seed

    def _one(self, k):
        x = np.asarray(self.X[k], np.float32)
        rng = np.random.default_rng(self.seed * 1_000_003 + k)
        return x[:, rng.permutation(x.shape[1])]


class ArrayView(_BaseView):
    """Plain wrapper so precomputed arrays and memmaps share one interface."""

    def _one(self, k):
        return np.asarray(self.X[k], np.float32)
