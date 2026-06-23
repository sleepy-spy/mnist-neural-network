from numpy.typing import NDArray
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

from corrupt import Corrupt
from utils import (
    ReLu,
    ReLu_differentiation,
    stable_softmax,
    one_hot_generation,
    cross_entropy_loss,
    dropout_mask,
)
from metrics import is_correct_output, wrong_counter
from display import epoch_details


class Layer:
    def __init__(self, fan_in: int, fan_out: int):
        self.weights = self.he_normal(fan_in, fan_out)
        self.bias = np.zeros(fan_out)

    def he_normal(self, fan_in: int, fan_out: int) -> NDArray[np.float64]:
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_in, fan_out)
        )


class LoadData:
    @classmethod
    def load_training(
        cls, filepath: str, val_split: float = 0.9
    ) -> tuple[
        tuple[NDArray[np.float64], NDArray[np.int_]],
        tuple[NDArray[np.float64], NDArray[np.int_]],
    ]:
        data = np.loadtxt(filepath, delimiter=",")
        X, y = data[:, 1:] / 255, data[:, 0].astype(int)
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        split = int(val_split * len(X))
        return (X[:split], y[:split]), (X[split:], y[split:])

    @classmethod
    def load_test(
        cls, filepath: str, has_labels: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.int_] | None]:
        data = np.loadtxt(filepath, delimiter=",")
        if has_labels:
            return data[:, 1:] / 255, data[:, 0].astype(int)
        return data / 255, None


class IO:
    @staticmethod
    def save_parameters(l1: Layer, l2: Layer, l3: Layer, filepath: str) -> None:
        parameters = {
            "layer 1 weights": l1.weights,
            "layer 1 bias": l1.bias,
            "layer 2 weights": l2.weights,
            "layer 2 bias": l2.bias,
            "layer 3 weights": l3.weights,
            "layer 3 bias": l3.bias,
        }
        np.savez(filepath, **parameters)

    @staticmethod
    def load_parameters(l1: Layer, l2: Layer, l3: Layer, filepath: str) -> None:
        try:
            data = np.load(filepath, allow_pickle=True)
            l1.weights = data["layer 1 weights"]
            l1.bias = data["layer 1 bias"]
            l2.weights = data["layer 2 weights"]
            l2.bias = data["layer 2 bias"]
            l3.weights = data["layer 3 weights"]
            l3.bias = data["layer 3 bias"]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing parameter {e} in the file.")

    @staticmethod
    def save_hyperparameters(
        lr: float,
        weight_decay: float,
        batch_size: int,
        dropout_prob: float,
        filepath: str,
    ) -> None:
        hyperparameters = {
            "learning rate": lr,
            "weight decay": weight_decay,
            "batch size": batch_size,
            "dropout prob": dropout_prob,
        }
        np.savez(filepath, **hyperparameters)

    @staticmethod
    def load_hyperparameters(filepath: str) -> dict:
        try:
            data = np.load(filepath, allow_pickle=True)
            return {
                "learning rate": float(data["learning rate"]),
                "weight decay": float(data["weight decay"]),
                "batch size": int(data["batch size"]),
                "dropout prob": float(data["dropout prob"]),
            }
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing hyperparameter {e} in the file.")
        return {}


class Network:
    def __init__(self):
        self.l1 = Layer(784, 1024)
        self.l2 = Layer(1024, 512)
        self.l3 = Layer(512, 10)
        self.lr = 0.08
        self.weight_decay = 0.001
        self.dropout_prob = 0.0
        self.batch_size = 64

    def forward_propagation(self, a0: NDArray[np.float64], dropout: bool) -> None:
        self.z1 = a0 @ self.l1.weights + self.l1.bias
        self.a1 = ReLu(self.z1)
        self.d1 = dropout_mask(self.a1, self.dropout_prob) if dropout else self.a1
        self.z2 = self.d1 @ self.l2.weights + self.l2.bias
        self.a2 = ReLu(self.z2)
        self.d2 = dropout_mask(self.a2, self.dropout_prob) if dropout else self.a2
        self.z3 = self.d2 @ self.l3.weights + self.l3.bias
        self.a3 = stable_softmax(self.z3)

    def backward_propagation(
        self, a0: NDArray[np.float64], correct_answer: NDArray[np.float64]
    ) -> None:
        clip_value = 1.0
        y_true = one_hot_generation(correct_answer)
        delta3 = self.a3 - y_true
        dW3 = np.clip(
            (self.a2.T @ delta3) / self.batch_size
            + self.weight_decay * self.l3.weights,
            -clip_value,
            clip_value,
        )
        db3 = np.clip(np.sum(delta3, axis=0) / self.batch_size, -clip_value, clip_value)
        delta2 = (delta3 @ self.l3.weights.T) * ReLu_differentiation(self.z2)
        dW2 = np.clip(
            (self.a1.T @ delta2) / self.batch_size
            + self.weight_decay * self.l2.weights,
            -clip_value,
            clip_value,
        )
        db2 = np.clip(np.sum(delta2, axis=0) / self.batch_size, -clip_value, clip_value)
        delta1 = (delta2 @ self.l2.weights.T) * ReLu_differentiation(self.z1)
        dW1 = np.clip(
            a0.T @ delta1 / self.batch_size + self.weight_decay * self.l1.weights,
            -clip_value,
            clip_value,
        )
        db1 = np.clip(np.sum(delta1, axis=0) / self.batch_size, -clip_value, clip_value)
        self.l3.weights -= self.lr * dW3
        self.l3.bias -= self.lr * db3
        self.l2.weights -= self.lr * dW2
        self.l2.bias -= self.lr * db2
        self.l1.weights -= self.lr * dW1
        self.l1.bias -= self.lr * db1


def train(training_file, save_weights) -> None:
    nn = Network()
    (X_train, y_train), (X_val, y_val) = LoadData.load_training(training_file)
    best_validation_accuracy = 0.0
    best_epoch = 0
    epoch = 0

    models_dir = Path(__file__).resolve().parent.parent / "models" / "untested"
    if save_weights:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = models_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        weights_path = str(run_dir / "model_parameters.npz")

    try:
        while True:
            epoch += 1
            total_epoch_loss = 0.0
            correct_outputs = 0
            for row in range(0, len(X_train), nn.batch_size):
                batch_X = X_train[row : row + nn.batch_size]
                batch_y = y_train[row : row + nn.batch_size]
                nn.forward_propagation(batch_X, True)
                total_epoch_loss += cross_entropy_loss(nn.a3, batch_y)
                nn.backward_propagation(batch_X, batch_y)
                correct_outputs += is_correct_output(nn.a3, batch_y)
            average_training_loss = total_epoch_loss / len(X_train)
            training_accuracy = correct_outputs / len(X_train)
            (
                average_validation_loss,
                validation_accuracy,
                validation_wrong_predictions,
            ) = validate(X_val, y_val, nn)
            epoch_details(
                epoch,
                average_training_loss,
                training_accuracy,
                average_validation_loss,
                validation_accuracy,
                validation_wrong_predictions,
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_epoch = epoch
                if save_weights:
                    IO.save_parameters(nn.l1, nn.l2, nn.l3, filepath=weights_path)
                    print(f"Parameters saved to {weights_path}")

            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
        print(f"Best epoch: {best_epoch}")
        print(f"Epoch {best_epoch} validation accuracy: {best_validation_accuracy}")


def validate(
    X_validation: NDArray[np.float64], y_validation: NDArray[np.int_], nn: Network
) -> tuple[float, float, dict[int, int]]:
    validation_wrong_predictions = {num_class: 0 for num_class in range(10)}
    correct_outputs = 0
    total_validation_loss = 0.0
    for row in range(0, len(X_validation), nn.batch_size):
        batch_X = X_validation[row : row + nn.batch_size]
        batch_y = y_validation[row : row + nn.batch_size]
        nn.forward_propagation(batch_X, False)
        total_validation_loss += cross_entropy_loss(nn.a3, batch_y)
        correct_outputs += is_correct_output(nn.a3, batch_y)
        validation_wrong_predictions = wrong_counter(
            nn.a3, batch_y, validation_wrong_predictions
        )
    return (
        total_validation_loss / len(X_validation),
        correct_outputs / len(X_validation),
        validation_wrong_predictions,
    )


def test(testing_file, weights_file) -> None:
    nn = Network()
    nn.batch_size = 512
    IO.load_parameters(nn.l1, nn.l2, nn.l3, filepath=weights_file)

    data = np.loadtxt(testing_file, delimiter=",")
    has_labels = data.shape[1] == 785
    if has_labels:
        X_test, y_test = data[:, 1:] / 255, data[:, 0].astype(int)
    else:
        X_test = data / 255

    predictions_path = str(Path(weights_file).parent / "predictions.csv")
    with open(predictions_path, "w") as f:
        if has_labels:
            f.write("ImageId,Label,TopGuessNLL,Entropy\n")
        else:
            f.write("ImageId,Label\n")

    with open(predictions_path, "a") as f_pred:
        for row in range(0, len(X_test), nn.batch_size):
            batch_X = X_test[row : row + nn.batch_size]
            nn.forward_propagation(batch_X, False)
            for i, preds in enumerate(nn.a3):
                image_id = row + i + 1
                pred = np.argmax(preds)
                if has_labels:
                    f_pred.write(
                        f"{image_id},{pred},"
                        f"{-np.log(np.clip(preds[pred], 1e-10, 1.0)):.6f},"
                        f"{-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))):.6f}\n"
                    )
                else:
                    f_pred.write(f"{image_id},{pred}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 neural_network.py train <training_file> [yes]")
        print("  python3 neural_network.py test <testing_file> <weights_file>")
        return

    command = sys.argv[1].lower()

    if command == "train":
        training_file = sys.argv[2]
        if len(sys.argv) >= 4 and sys.argv[3].lower() == "yes":
            save_weights = True
        else:
            save_weights = False
        train(training_file, save_weights)

    elif command == "test":
        if len(sys.argv) < 4:
            print("Usage: python3 neural_network.py test <testing_file> <weights_file>")
            return
        testing_file = sys.argv[2]
        weights_file = sys.argv[3]
        test(testing_file, weights_file)

    else:
        print(f"Unknown command: {command}")
        print("Use 'train' or 'test'")


if __name__ == "__main__":
    main()
