# MNIST Neural Network — Project History & Plan

## Project overview
An MNIST digit classifier built entirely with Python 3 + NumPy (no ML libraries like PyTorch or TensorFlow). Uses a fully connected network with He Normal initialization. Also contains data corruption utilities for experiments.

## What has been done

### Core neural network
- **Three-layer network**: 784 → 1024 → 512 → 10, with ReLU hidden activations, softmax output, cross-entropy loss
- **Training loop**: mini-batch gradient descent with configurable batch size, learning rate, weight decay, and dropout
- **Validation & testing**: epoch-wise validation, test inference with predictions written to CSV
- Wired `test_details()` into `test()` — tracks accuracy, per-class wrong predictions, and total test loss when labels exist
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
├── display.py         # epoch_details, test_details
├── metrics.py         # is_correct_output, wrong_counter
├── neural_network.py  # Layer, IO, LoadData, Network, train, validate, test, main
├── utils.py           # ReLu, stable_softmax, one_hot, cross_entropy_loss, dropout_mask
└── visualise.py       # visualization (takes predictions CSV as CLI arg)
```

### Known issues (open)
- **`save_hyperparameters` / `load_hyperparameters`** in `IO` class — never called. Dead code. Consider removing.
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

**Last commit**: `feat: wire test_details into test function`
