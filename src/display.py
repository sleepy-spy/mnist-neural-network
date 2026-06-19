def epoch_details(
    epoch_num: int,
    training_loss: float,
    training_accuracy: float,
    validation_loss: float,
    validation_accuracy: float,
    validation_wrong_predictions: dict[int, int],
) -> None:
    print("-" * 50)
    print(f"Epoch {epoch_num}")
    print()
    print(f"Average training loss: {training_loss}")
    print(f"Training accuracy: {training_accuracy}")
    print(f"Average validation loss: {validation_loss}")
    print(f"Validation accuracy: {validation_accuracy}")
    print(
        "The following are the number of times the wrong class was predicted for each class:"
    )
    print()
    for num_class, wrong_predictions in validation_wrong_predictions.items():
        print(f"{num_class}: {wrong_predictions}")
    print("-" * 50)


def test_details(
    testing_accuracy: float,
    test_wrong_predictions: dict[int, int],
) -> None:
    print("-" * 50)
    print(f"Testing accuracy: {testing_accuracy}")
    print()
    print(
        "The following are the number of times the wrong class was predicted for each class:"
    )
    print()
    for num_class, wrong_predictions in test_wrong_predictions.items():
        print(f"{num_class}: {wrong_predictions}")
    print("-" * 50)
