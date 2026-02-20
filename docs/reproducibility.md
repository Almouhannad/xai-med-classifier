# Reproducibility Notes

## Training Loop Structure

Training now uses explicit, modular epoch loops:

- `train_one_epoch(...)` for forward/backward/update on training batches.
- `validate_one_epoch(...)` for evaluation-only validation batches.
- `run_training(...)` as the orchestrator that wires model, optimizer, dataloaders, metrics logging, and checkpoint persistence.

These functions live in `src/xaimed/train/loops.py` and `src/xaimed/train/train.py`.

## Checkpointing

Each run writes:

- `last.pt`: checkpoint from the most recent epoch.
- `best.pt`: checkpoint with the lowest validation loss observed so far.

Both files contain model/optimizer state dicts, epoch index, and train/validation metrics.

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
- Computes aggregate metrics (`accuracy`, `macro_f1`, `num_samples`).
- Saves `metrics.json` and an annotated `confusion_matrix.png` in `eval.output_dir`.

Smoke defaults (in `configs/experiments/quick_smoke.yaml`):

- `eval.checkpoint_path: artifacts/checkpoints/quick_smoke/best.pt`
- `eval.output_dir: artifacts/eval/quick_smoke`
- `eval.device: cpu`
