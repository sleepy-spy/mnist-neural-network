import numpy as np
from numpy.typing import NDArray


def ReLu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)


def ReLu_differentiation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x > 0).astype(float)


def stable_softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot_generation(
    correct_answers: NDArray[np.int_], num_classes: int = 10
) -> NDArray[np.float64]:
    y_true = np.zeros((correct_answers.shape[0], num_classes))
    y_true[np.arange(len(correct_answers)), correct_answers] = 1.0
    return y_true


def cross_entropy_loss(
    a3: NDArray[np.float64], correct_answers: NDArray[np.int_]
) -> float:
    y_pred = np.clip(a3, 1e-10, 1.0)
    correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
    return -np.sum(np.log(correct_probs))


def dropout_mask(
    activated_output: NDArray[np.float64], dropout_prob: float
) -> NDArray[np.float64]:
    mask = (np.random.rand(*activated_output.shape) > dropout_prob).astype(float)
    return activated_output * mask
