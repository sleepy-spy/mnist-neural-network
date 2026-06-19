import numpy as np
from numpy.typing import NDArray


def is_correct_output(
    a3: NDArray[np.float64], correct_answers: NDArray[np.int_]
) -> int:
    predicted_classes = np.argmax(a3, axis=1)
    return int(np.sum(predicted_classes == correct_answers))


def wrong_counter(
    a3: NDArray[np.float64],
    correct_answers: NDArray[np.int_],
    wrong_predictions: dict[int, int],
) -> dict[int, int]:
    predicted_classes = np.argmax(a3, axis=1)
    correct_mask = predicted_classes == correct_answers
    for true_class in correct_answers[~correct_mask]:
        wrong_predictions[int(true_class)] += 1
    return wrong_predictions
