# Reproducibility Notes

## Training Loop Structure

Training now uses explicit, modular epoch loops:

- `train_one_epoch(...)` for forward/backward/update on training batches.
- `validate_one_epoch(...)` for evaluation-only validation batches.
- `run_training(...)` as the orchestrator that wires model, optimizer, dataloaders, metrics logging, and checkpoint persistence.

These functions live in `src/xaimed/train/loops.py` and `src/xaimed/train/train.py`.

To reduce run-to-run variance, training/evaluation now seed Python, NumPy, and Torch RNGs via `xaimed.seed.set_global_seed(...)` using `config.seed` (default `42`).

## Checkpointing

Each run writes:

- `last.pt`: checkpoint from the most recent epoch.
- `best.pt`: checkpoint with the best monitored validation score (based on `early_stopping_monitor` + `early_stopping_mode`).

Both files contain model/optimizer state dicts, epoch index, train/validation metrics, history, and scheduler state (when a scheduler is enabled).


## Tunable Training Hyperparameters

The `train` config supports optimization and stopping controls:

- Optimizer: `optimizer` (`adam`, `adamw`, `sgd`) with `lr`, `weight_decay`, and `momentum`/`nesterov` for SGD.
- LR scheduler: `lr_scheduler` (`none`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`) and its related parameters (`lr_step_size`, `lr_gamma`, `lr_t_max`, `lr_eta_min`, `lr_plateau_*`) including `lr_plateau_monitor` (`val_loss`/`val_accuracy`).
- Gradient clipping: `grad_clip_norm` to clip gradient norm each training step.
- Early stopping: `early_stopping`, `early_stopping_patience`, `early_stopping_min_delta`, `early_stopping_monitor` (`val_loss`/`val_accuracy`), and `early_stopping_mode` (`min`/`max`).

The CLI now prints `best_epoch`, `epochs_ran`, and whether early stopping triggered at the end of training.

## Smoke Training

`make train-smoke` is configured to run fully on CPU with synthetic data to keep CI/runtime stable and fast.

- Config: `configs/experiments/quick_smoke.yaml`
- Device: `cpu`
- Epochs: `1`
- Data mode: `use_fake_data: true`

This validates the end-to-end trainer and checkpoint writing logic without requiring dataset downloads.


## Evaluation & Confusion Matrix

The evaluation pipeline is available via `xaimed eval` and `make eval-smoke`.

- Loads the trained checkpoint (default: `train.checkpoint_dir/best.pt`).
- Rebuilds the configured dataloader split (default: `val`).
- Computes aggregate metrics (`accuracy`, `macro_f1`, `ece`, `num_samples`).
- Saves `metrics.json`, `confusion_matrix.png`, and `training_curves.png` in `eval.output_dir`.
- Exports failure-analysis artifacts: `failure_gallery_selection.csv`, `high_confidence_wrongs_grid.png`, and `low_confidence_corrects_grid.png`.

Smoke defaults (in `configs/experiments/quick_smoke.yaml`):

- `eval.checkpoint_path: artifacts/checkpoints/quick_smoke/best.pt`
- `eval.output_dir: artifacts/eval/quick_smoke`
- `eval.device: cpu`


## Config Coverage

Both configs (`configs/default.yaml` and `configs/experiments/quick_smoke.yaml`) define `train`, `data`, `eval`, and `report` sections to keep CLI behavior consistent across full and smoke runs.
