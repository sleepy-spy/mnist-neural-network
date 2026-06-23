# MNIST Neural Network — Project History & Plan

## Project overview
An MNIST digit classifier built entirely with Python 3 + NumPy (no ML libraries like PyTorch or TensorFlow). Uses a fully connected network with He Normal initialization. Also contains data corruption utilities for experiments.

## What has been done

### Core neural network
- **Three-layer network**: 784 → 1024 → 512 → 10, with ReLU hidden activations, softmax output, cross-entropy loss
- **Training loop**: mini-batch gradient descent with configurable batch size, learning rate, weight decay, and dropout
- **Validation & testing**: epoch-wise validation, test inference with predictions written to CSV
- **Weight saving/loading**: IO class saves/loads model parameters as `.npz` files
- **Model weights stored at** `models/0.97814/model_kaggle_parameters.npz` (97.8% accuracy)

### Refactoring & file structure
- Monolithic `Neural_Network.py` was broken into `src/neural_network.py`, `src/utils.py`, `src/metrics.py`, `src/display.py`, `src/corrupt.py`
- Pure functions (ReLU, softmax, one-hot encoding, cross-entropy loss, dropout) extracted to `src/utils.py`
- Static `IO` class added for saving/loading parameters and hyperparameters
- Fixed bugs in `train()` (crash on first epoch, shuffling per epoch, weight save paths), `test()` (signature mismatch, pixel clipping, file I/O), and `validate()` (variable name errors)
- `src/visualise.py` fixed: `LoadData` constructor → `load_test()` classmethod, predictions path via `sys.argv[1]`, import fixed, unused imports removed, `Y_test` → `y_test`
- Model save path lowercased: `"Models"/"Untested"` → `"models"/"untested"`
- Predictions CSV column `Loss` renamed to `TopGuessNLL` (no true label — just `-log(p)` of top guess)

### Current file layout
```
src/
├── corrupt.py         # Corrupt class (label/feature corruption)
├── display.py         # epoch_details, test_details (test_details is unused)
├── metrics.py         # is_correct_output, wrong_counter
├── neural_network.py  # Layer, IO, LoadData, Network, train, validate, test, main
├── utils.py           # ReLu, stable_softmax, one_hot, cross_entropy_loss, dropout_mask
└── visualise.py       # visualization (takes predictions CSV as CLI arg)
```

### Known issues (open)
- **`test_details()`** in `display.py` — never called. Wiring into `test()` is drafted below but not yet implemented.

  Proposed full `test()` function with accuracy tracking:
  ```python
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

      test_wrong_predictions = {num_class: 0 for num_class in range(10)}
      correct_outputs = 0
      total_test_loss = 0.0

      with open(predictions_path, "w") as f:
          if has_labels:
              f.write("ImageId,Label,TopGuessNLL,Entropy\n")
          else:
              f.write("ImageId,Label\n")

      with open(predictions_path, "a") as f_pred:
          for row in range(0, len(X_test), nn.batch_size):
              batch_X = X_test[row : row + nn.batch_size]
              if has_labels:
                  batch_y = y_test[row : row + nn.batch_size]
              nn.forward_propagation(batch_X, False)
              if has_labels:
                  total_test_loss += cross_entropy_loss(nn.a3, batch_y)
                  correct_outputs += is_correct_output(nn.a3, batch_y)
                  test_wrong_predictions = wrong_counter(
                      nn.a3, batch_y, test_wrong_predictions
                  )
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

      if has_labels:
          testing_accuracy = correct_outputs / len(X_test)
          test_details(testing_accuracy, test_wrong_predictions)
  ```

  Also update import (line 17): `from display import epoch_details` → `from display import epoch_details, test_details`

- **`save_hyperparameters` / `load_hyperparameters`** in `IO` class — never called. Dead code.
- **`src/Neural_Network.py`** — old monolith still present. Consider removing.

## What we're planning next

### Goal
Build a web frontend where a user can draw a digit and get a real-time prediction from the trained model.

### Approach: Flask + HTML Canvas + JavaScript
Chosen over Gradio/Streamlit to learn the web stack (HTML, CSS, JavaScript, HTTP, Flask).

### What it teaches
1. **HTML** — `<canvas>`, page structure, element attributes, forms
2. **CSS** — layout, dark theme, styling canvas and buttons
3. **JavaScript** — canvas 2D drawing (mouse events), getting pixel data, `fetch()` API for HTTP requests, async/await, JSON
4. **Flask** — routes, serving HTML, receiving POST data, returning JSON
5. **Image processing** — using Pillow to convert RGBA → grayscale, resize 280×280 → 28×28, normalize for model input

### Files to create
- **`templates/index.html`** — the drawing interface (canvas, buttons, result display, inline CSS + JS)
- **`src/app.py`** — Flask server with two routes: `GET /` serves the page, `POST /predict` processes the image and runs inference

### Pipeline
Canvas (280×280 RGBA) → Pillow resize to 28×28 → grayscale → normalize `/ 255` → reshape to `(1, 784)` → `Network.forward_propagation()` → `np.argmax()` → return JSON `{ digit, confidence }`

### Dependencies
- `Flask` (to be installed)

### Canvas UX
Black background, white pen (matches MNIST natively — no color inversion needed).

---

**Last commit**: `359a293 docs: add project history and frontend plan to HANDOFF.md`
