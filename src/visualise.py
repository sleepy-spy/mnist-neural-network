import matplotlib.pyplot as plt
import numpy as np
from neural_network import LoadData
from random import randint
big_data = LoadData(test_label=True)
X_test, Y_test = big_data.load_test_data()
predictions = np.loadtxt(
        "./Models/0.97814/Model Results/MNIST Kaggle Test Set/submissions.csv",
        delimiter=",",
        skiprows=1,
    ) 

def display_random_misclassified_digits():
    correct_mask = predictions[:, 1] == Y_test[:]
    wrong_predictions = predictions[~correct_mask, 1]
    wrong_images = X_test[~correct_mask, :]
    wrong_images_true_label = Y_test[~correct_mask]
    permuted_indices = np.random.permutation(wrong_predictions.shape[0])
    wrong_predictions, wrong_images, wrong_images_true_label = (
        wrong_predictions[permuted_indices],
        wrong_images[permuted_indices],
        wrong_images_true_label[permuted_indices],
    )
    if len(wrong_images):
        plt.figure(figsize=(10, 10))
        num_to_show = min(25, len(wrong_images))
        for i in range(num_to_show):
            try:
                plt.subplot(5, 5, i + 1)
                plt.imshow(wrong_images[i, :].reshape((28, 28)), cmap="binary")
                plt.title(
                    f"Pred: {int(wrong_predictions[i])}\nTrue: {int(wrong_images_true_label[i])}",
                    fontsize=8,
                )
                plt.axis("off")
            except IndexError:
                pass
        plt.tight_layout()
        plt.show()
    else:
        print("Everything correct! 100% model accuracy!")

def display_ugliest_digits():
    ugliest_indices = np.argsort(-predictions[:, 3])
    ugliest_images = X_test[ugliest_indices, :]
    ugliest_images_label = Y_test[ugliest_indices]
    if len(ugliest_images):
        plt.figure(figsize=(10, 10))
        num_to_show = min(25, len(ugliest_images))
        for i in range(num_to_show):
            try:
                plt.subplot(5, 5, i + 1)
                plt.imshow(ugliest_images[i, :].reshape((28, 28)), cmap="binary")
                plt.title(f"True: {int(ugliest_images_label[i])}", fontsize=8)
                plt.axis("off")
            except IndexError:
                pass
        plt.tight_layout()
        plt.show()
    else:
        print("Nothing to see here!")
    
display_ugliest_digits()
