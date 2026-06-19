# Handoff — Session 3

## What was done

### File restructuring
- **`Neural_Network.py`** renamed to `neural_network.py` (case-insensitive filesystem, so git sees it as the same file)
- **`Corrupt`** class extracted to `src/corrupt.py`
- **`Layer`** class stays in `neural_network.py`
- **Pure functions extracted:**
  - `src/utils.py` — `ReLu`, `ReLu_differentiation`, `stable_softmax`, `one_hot_generation`, `cross_entropy_loss`, `dropout_mask`
  - `src/metrics.py` — `is_correct_output`, `wrong_counter`
  - `src/display.py` — `epoch_details`, `test_details` (dead code, never called)
- **`IO` static class** added to `neural_network.py` — `save_parameters`, `load_parameters`, `save_hyperparameters`, `load_hyperparameters`

### Bugs fixed
1. **`train()`** — was crashing on first epoch (`big_data` didn't exist). Now:
   - Calls `LoadData.load_training(training_file)` once before epoch loop
   - Uses `while True` with manual `epoch += 1` instead of `for epoch in range(max_epochs)`
   - Passes `(X_val, y_val)` to `validate()` directly
   - Shuffles training data every epoch via `np.random.permutation`
   - Saves weights to `Models/Untested/<timestamp>/model_parameters.npz` (unique per run)
   - `save_weights` used as bool (was `== "y"`)
2. **`test()`** — was crashing (signature mismatch, old `big_data` API). Now:
   - Signature: `test(testing_file, weights_file)` — matches `main()`
   - Auto-detects labels by column count (785 = has labels)
   - Flat file I/O (no reopen-per-batch)
   - Fixed `np.clip(prediction[pred], 1e-10, 1.0)` (was missing `a_min`/`a_max`)
   - Writes predictions to same dir as weights file
   - No labels CSV output (redundant — labels are in the original file)
3. **`validate()`** — `Y_validation` → `y_validation`, `small_data`/`correct_answers` → `batch_X`/`batch_y`

### File layout
```
src/
├── corrupt.py         # Corrupt class (corruption utilities)
├── display.py         # epoch_details, test_details (printing)
├── metrics.py         # is_correct_output, wrong_counter
├── neural_network.py  # Layer, IO, LoadData, Network, train, validate, test, main
├── utils.py           # ReLu, stable_softmax, one_hot, cross_entropy_loss, dropout_mask
└── visualise.py       # visualization (still broken — see below)
```

## What's still left

### Known issues
- **`visualise.py`** — still calls `LoadData(test_label=True)` with non-existent constructor. Needs migration to `LoadData.load_test()`. Also references old predictions path `./Models/0.97814/Model Results/...`.
- **`test_details()`** — still exists in `display.py` but never called anywhere. Dead code.
- **`save_hyperparameters` / `load_hyperparameters`** — in `IO` class, never called. Dead code.

### Commits
```
870b55c refactor: restructure Network class, fix train/test bugs, and reorganize file layout
```
