from numpy.typing import NDArray
import numpy as np


class Corrupt:
    @staticmethod
    def corrupt_labels(
        y: NDArray[np.float64], corruption_rate=1.0
    ) -> NDArray[np.float64]:
        mask = np.random.rand(len(y)) < corruption_rate
        y_corrupted = y.copy()
        y_corrupted[mask] = np.random.randint(0, 10, size=np.sum(mask))
        return y_corrupted

    @staticmethod
    def corrupt_features_gaussian(
        x: NDArray[np.float64], corruption_rate=0.67
    ) -> NDArray[np.float64]:
        mask = np.random.rand(*x.shape) < corruption_rate
        x_corrupted = x.copy()
        x_corrupted[mask] = x[mask] + np.random.normal(0, 0.5, x[mask].shape)
        x_corrupted = np.clip(x_corrupted, 0, 1)
        return x_corrupted

    @staticmethod
    def corrupt_features_uniform(
        x: NDArray[np.float64], corruption_rate=1.0
    ) -> NDArray[np.float64]:
        mask = np.random.rand(len(x)) < corruption_rate
        x_corrupted = x.copy()
        x_corrupted[mask] = np.random.rand(*x_corrupted[mask].shape)
        return x_corrupted
