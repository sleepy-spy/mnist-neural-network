from numpy.typing import NDArray
import numpy as np
import sys


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


class Layer:
    def __init__(self, fan_in: int, fan_out: int):
        self.weights = self.he_normal(
            fan_in, fan_out
        )  # The weight matrix is of the order n_in * n_out
        self.bias = np.zeros(
            fan_out
        )  # The bias vector is a column vector with n_out biases

    def he_normal(self, fan_in: int, fan_out: int) -> NDArray[np.float64]:
        return np.random.normal(
            loc=0.0, scale=np.sqrt(2 / fan_in), size=(fan_in, fan_out)
        )


class Network:
    def __init__(self):
        self.l1 = Layer(784, 1024)
        self.l2 = Layer(1024, 512)
        self.l3 = Layer(512, 10)
        self.lr = 0.08
        self.weight_decay = 0.001
        self.dropout_prob = 0.0
        self.batch_size = 64

    def dropout_mask(
        self, activated_output: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        mask = (np.random.rand(*activated_output.shape) > self.dropout_prob).astype(
            float
        )
        dropped_output = activated_output * mask
        return dropped_output

    def forward_propagation(self, a0: NDArray[np.float64], dropout: bool) -> None:
        self.z1 = a0 @ self.l1.weights + self.l1.bias
        self.a1 = self.ReLu(self.z1)
        self.d1 = self.dropout_mask(self.a1) if dropout else self.a1
        self.z2 = self.d1 @ self.l2.weights + self.l2.bias
        self.a2 = self.ReLu(self.z2)
        self.d2 = self.dropout_mask(self.a2) if dropout else self.a2
        self.z3 = self.d2 @ self.l3.weights + self.l3.bias
        self.a3 = self.stable_softmax(self.z3)

    def stable_softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward_propagation(
        self, a0: NDArray[np.float64], correct_answer: NDArray[np.float64]
    ) -> None:
        clip_value = 1.0
        y_true = self.one_hot_generation(correct_answer)
        delta3 = self.a3 - y_true
        dW3 = np.clip(
            (self.a2.T @ delta3) / self.batch_size
            + self.weight_decay * self.l3.weights,
            -clip_value,
            clip_value,
        )
        db3 = np.clip(np.sum(delta3, axis=0) / self.batch_size, -clip_value, clip_value)
        delta2 = (delta3 @ self.l3.weights.T) * self.ReLu_differentiation(self.z2)
        dW2 = np.clip(
            (self.a1.T @ delta2) / self.batch_size
            + self.weight_decay * self.l2.weights,
            -clip_value,
            clip_value,
        )
        db2 = np.clip(np.sum(delta2, axis=0) / self.batch_size, -clip_value, clip_value)
        delta1 = (delta2 @ self.l2.weights.T) * self.ReLu_differentiation(self.z1)
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

    def one_hot_generation(
        self, correct_answers: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        y_true = np.zeros((correct_answers.shape[0], 10))
        y_true[np.arange(len(correct_answers)), correct_answers.astype(int)] = 1.0
        return y_true

    def ReLu_differentiation(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        return (array > 0).astype(float)

    def ReLu(self, pre_activation_output: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.maximum(0, pre_activation_output)

    def cross_entropy_loss(self, correct_answers: NDArray[np.float64]) -> float:
        y_pred = np.clip(self.a3, 1e-10, 1.0)
        correct_probs = y_pred[np.arange(len(y_pred)), correct_answers]
        return -np.sum(np.log(correct_probs))

    def epoch_details(
        self,
        epoch_num: int,
        training_loss: float,
        training_accuracy: float,
        validation_loss: float,
        validation_accuracy: float,
        validation_wrong_predictions: dict[int, int],
    ) -> None:
        print("-" * 50)
        print(f"Epoch {epoch_num+1}")
        print()
        print(f"Average training loss: {training_loss}")
        print(f"Training accuracy: {training_accuracy}")
        print(f"Average validation loss: {validation_loss}")
        print(f"Validation accuracy: {validation_accuracy}")
        print(
            f"The following are the number of times the wrong class was predicted for each class:"
        )
        print()
        for num_class, wrong_predictions in validation_wrong_predictions.items():
            print(f"{num_class}: {wrong_predictions}")
        print("-" * 50)

    def test_details(
        self, testing_accuracy: float, test_wrong_predictions: dict[int, int]
    ) -> None:
        print("-" * 50)
        print(f"Testing accuracy: {testing_accuracy}")
        print()
        print(
            f"The following are the number of times the wrong class was predicted for each class:"
        )
        print()
        for num_class, wrong_predictions in test_wrong_predictions.items():
            print(f"{num_class}: {wrong_predictions}")
        print("-" * 50)

    def is_correct_output(self, correct_answers: NDArray[np.float64]) -> int:
        predicted_classes = np.argmax(self.a3, axis=1)
        correct_mask = predicted_classes == correct_answers
        return np.sum(correct_mask)

    def wrong_counter(
        self, correct_answers: NDArray[np.float64], wrong_predictions: dict[int, int]
    ) -> dict[int, int]:
        predicted_classes = np.argmax(self.a3, axis=1)
        correct_mask = predicted_classes == correct_answers
        for true_class in correct_answers[~correct_mask]:
            wrong_predictions[true_class] += 1
        return wrong_predictions

    def save_parameters(self, filepath: str) -> None:
        parameters = {
            "layer 1 weights": self.l1.weights,
            "layer 1 bias": self.l1.bias,
            "layer 2 weights": self.l2.weights,
            "layer 2 bias": self.l2.bias,
            "layer 3 weights": self.l3.weights,
            "layer 3 bias": self.l3.bias,
        }
        np.savez(filepath, **parameters)

    def save_hyperparameters(self, filepath: str) -> None:
        hyperparameters = {
            "learning rate": self.lr,
            "weight decay": self.weight_decay,
            "batch size": self.batch_size,
            "dropout prob": self.dropout_prob,
        }
        np.savez(filepath, **hyperparameters)

    def load_parameters(self, filepath: str) -> None:
        try:
            load_parameters = np.load(filepath, allow_pickle=True)
            self.l1.weights = load_parameters["layer 1 weights"]
            self.l1.bias = load_parameters["layer 1 bias"]
            self.l2.weights = load_parameters["layer 2 weights"]
            self.l2.bias = load_parameters["layer 2 bias"]
            self.l3.weights = load_parameters["layer 3 weights"]
            self.l3.bias = load_parameters["layer 3 bias"]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing parameter {e} in the file.")

    def load_hyperparameters(self, filepath: str) -> None:
        try:
            load_hyperparameters = np.load(filepath, allow_pickle=True)
            self.lr = load_hyperparameters["learning rate"]
            self.weight_decay = load_hyperparameters["weight decay"]
            self.batch_size = load_hyperparameters["batch size"]
            self.dropout_prob = load_hyperparameters["dropout prob"]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
        except KeyError as e:
            print(f"Error: Missing hyperparameter {e} in the file.")


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


def train(big_data: LoadData) -> None:
    save_weights = (
        input("Do you wish to save the weights of the training? (y, n): ")
        .strip()
        .lower()
    )
    nn = Network()
    max_epochs = 2**63 - 1
    best_validation_accuracy = 0.0
    best_epoch = 0
    try:
        for epoch in range(max_epochs):
            X_train, Y_train = big_data.load_training_data()
            total_epoch_loss = 0.0
            correct_outputs = 0
            for row in range(0, len(X_train), nn.batch_size):
                small_data = X_train[row : row + nn.batch_size]
                correct_answers = Y_train[row : row + nn.batch_size]
                nn.forward_propagation(small_data, True)
                total_epoch_loss += nn.cross_entropy_loss(correct_answers)
                nn.backward_propagation(small_data, correct_answers)
                correct_outputs += nn.is_correct_output(correct_answers)
            average_training_loss = total_epoch_loss / len(X_train)
            training_accuracy = correct_outputs / len(X_train)
            (
                average_validation_loss,
                validation_accuracy,
                validation_wrong_predictions,
            ) = validate(*big_data.load_validation_data(), nn)
            nn.epoch_details(
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
                if save_weights == "y":
                    nn.save_parameters(
                        filepath="./Models/Untested/model_kaggle_parameters.npz"
                    )
                    print("Parameters saved")
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
        print(f"Best epoch: {best_epoch}")
        print(f"Epoch {best_epoch} validation accuracy: {best_validation_accuracy}")


def validate(
    X_validation: NDArray[np.float64], Y_validation: NDArray[np.float64], nn: Network
) -> tuple[float, float, dict[int, int]]:
    validation_wrong_predictions = {num_class: 0 for num_class in range(10)}
    correct_outputs = 0
    total_validation_loss = 0.0
    for row in range(0, len(X_validation), nn.batch_size):
        small_data = X_validation[row : row + nn.batch_size]
        correct_answers = Y_validation[row : row + nn.batch_size]
        nn.forward_propagation(small_data, False)
        total_validation_loss += nn.cross_entropy_loss(correct_answers)
        correct_outputs += nn.is_correct_output(correct_answers)
        validation_wrong_predictions = nn.wrong_counter(
            correct_answers, validation_wrong_predictions
        )
    return (
        total_validation_loss / len(X_validation),
        correct_outputs / len(X_validation),
        validation_wrong_predictions,
    )


def test(
    big_data: LoadData, parameter_path: str, predictions_path: str, labels_path: str
) -> None:
    nn = Network()
    nn.batch_size = 512
    nn.load_parameters(filepath=parameter_path)
    if big_data.label_present:
        X_test, Y_test = big_data.load_test_data()
    else:
        X_test = big_data.load_test_data()
    if big_data.label_present:
        with open(labels_path, "w") as f_labels:
            f_labels.write("ImageId,CorrectLabel\n")
        with open(predictions_path, "w") as f_predictions:
            f_predictions.write("ImageId,Label,Loss,Uncertainty\n")
    else:
        with open(predictions_path, "w") as f_predictions:
            f_predictions.write("ImageId,Label\n")
    for row in range(0, len(X_test), nn.batch_size):
        small_data = X_test[row : row + nn.batch_size]
        nn.forward_propagation(small_data, False)
        if big_data.label_present:
            correct_labels = Y_test[row : row + nn.batch_size]
            with open(
                predictions_path,
                "a",
            ) as f_predictions:
                for index, predictions in enumerate(nn.a3):
                    np.savetxt(
                        f_predictions,
                        np.array(
                            [
                                [
                                    index + row + 1,
                                    np.argmax(predictions),
                                    -np.log(
                                        np.clip(predictions[np.argmax(predictions)])
                                    ),
                                    -np.sum(
                                        predictions
                                        * np.log(np.clip(predictions, 1e-10, 1.0))
                                    ),
                                ]
                            ],
                        ),
                        fmt="%d,%d,%f,%f",
                        delimiter=",",
                    )

            with open(labels_path, "a") as f_labels:
                for index, correct_answer in enumerate(correct_labels):
                    np.savetxt(
                        f_labels,
                        np.array([[index + row + 1, correct_answer]]),
                        fmt="%d",
                        delimiter=",",
                    )
        else:
            with open(
                predictions_path,
                "a",
            ) as f_predictions:
                for predictions in nn.a3:
                    np.savetxt(
                        f_predictions,
                        np.array([[row + 1, np.argmax(predictions)]]),
                        fmt="%d",
                        delimiter=",",
                    )


def main():
    train_or_test = (
        input("Would you like to train the model or test the model? (train/test): ")
        .lower()
        .strip()
    )
    if train_or_test == "test":
        invalid_label_presence = True
        while invalid_label_presence:
            test_labels_presence = input("Does your test data have labels? (y/n): ")
            if test_labels_presence == "y":
                test_label = True
                invalid_label_presence = False
            elif test_labels_presence == "n":
                test_label = False
                invalid_label_presence = False
        big_data = LoadData(sys.argv[1], sys.argv[2], test_label=test_label)
        test(big_data=big_data)
        print("Testing complete")
    elif train_or_test == "train":
        big_data = LoadData(sys.argv[1], sys.argv[2], test_label=False)
        train(big_data=big_data)
        print("Training complete")


if __name__ == "__main__":
    main()
