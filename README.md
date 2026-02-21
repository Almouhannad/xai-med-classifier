# Explainable Medical Classifier (XAI-first)

A high-performance classification pipeline for medical imaging using **MedMNIST**, designed with explainability as a first-class citizen.

## Quickstart

1. **Setup Environment**:
   ```bash
   make setup
   ```

2. **Download Dataset**:
   ```bash
   python scripts/download_data.py --dataset pathmnist --data-dir data
   ```

3. **Run Smoke Test (Training)**:
   ```bash
   make train-smoke
   ```

4. **Run Smoke Evaluation (metrics + confusion matrix)**:
   ```bash
   make eval-smoke
   ```

5. **Generate Explanations (Grad-CAM overlays)**:
   ```bash
   make explain-smoke
   ```

6. **Generate Report**:
   ```bash
   make report
   ```


### Script entrypoints

All scripts in `scripts/` delegate to the CLI and support argument forwarding:

- `python scripts/download_data.py ...`
- `python scripts/train.py ...`
- `python scripts/eval.py ...`
- `python scripts/explain.py ...`
- `python scripts/make_report.py ...`

## Data Module

The MedMNIST data integration lives in `src/xaimed/data/medmnist.py` and provides:
- `build_transforms(image_size=224)` for standard RGB resize/tensor/normalize preprocessing.
- `load_medmnist_dataset(...)` for split-aware dataset construction.
- `build_medmnist_dataloaders(...)` for train/val/test loaders.
- `download_medmnist(...)` for split downloads and size reporting.

You can also download data via the CLI module:

```bash
python -m xaimed.cli download-data --dataset pathmnist --data-dir data
```

## Model Factory

Model construction is centralized in `src/xaimed/models/factory.py` via `build_model(...)`.

Current support:
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`

The factory automatically replaces the classifier head to match `num_classes` and can adapt the first convolution for non-RGB inputs (`in_channels != 3`). During channel adaptation, conv1 weights are remapped from pretrained RGB kernels: averaged for 1-channel inputs, and repeat/truncate + scale (`3 / in_channels`) for multi-channel inputs. This preserves pretrained signal while keeping early-layer activation magnitude closer to the RGB baseline.

For weights, `build_model(...)` supports either `pretrained=True` (uses torchvision `DEFAULT` weights) or an explicit `weights` argument (enum value or enum-name string such as `"DEFAULT"`). Note: when using pretrained weights, preprocessing should follow the corresponding torchvision weight transforms (`weights.transforms()`).



## Training & Checkpointing

Training is implemented in modular loops under `src/xaimed/train/`:

- `train_one_epoch(...)` and `validate_one_epoch(...)` encapsulate per-epoch logic.
- `run_training(...)` coordinates seed setup, model/optimizer wiring, loop execution, and checkpoint writing.

Checkpoints are saved as:

- `best.pt` (lowest validation loss)
- `last.pt` (latest epoch)

The smoke config (`configs/experiments/quick_smoke.yaml`) uses CPU + synthetic data so `make train-smoke` runs reliably in constrained environments.


## Evaluation

Evaluation is implemented in `src/xaimed/eval/evaluate.py` and exposed through the CLI:

```bash
python -m xaimed.cli --config configs/experiments/quick_smoke.yaml eval
```

Outputs are written under the configured `eval.output_dir` and include:
- `metrics.json` with `accuracy`, `macro_f1`, `ece`, sample count, and split.
- `confusion_matrix.png` containing an annotated confusion matrix heatmap.
- `failure_gallery_selection.csv` with two ranked groups: high-confidence wrong predictions and low-confidence correct predictions.
- `high_confidence_wrongs_grid.png` and `low_confidence_corrects_grid.png` for visual failure/success triage.

The smoke command `make eval-smoke` reads `configs/experiments/quick_smoke.yaml` and writes artifacts to `artifacts/eval/quick_smoke/`.

## Explainability (Grad-CAM)

Generate Grad-CAM overlays from a trained checkpoint:

```bash
python -m xaimed.cli --config configs/experiments/quick_smoke.yaml explain
```

Explain artifacts are written to the configured `explain.output_dir` (default quick smoke: `artifacts/explain/quick_smoke/`) as `sample_*_gradcam.png` files.

## Project Overview

This repository provides a standardized workflow for:
- Training deep learning models on MedMNIST datasets.
- Evaluating performance with robust metrics (AUC, Accuracy, F1).
- Applying XAI techniques like Grad-CAM and Integrated Gradients.
- Generating automated reports for model transparency.

## Ethics & Limitations
- This tool is for research purposes only and not intended for clinical diagnosis.
- Explainability methods (Grad-CAM, IG) are approximations and should be interpreted with caution.
- Data is sourced from [MedMNIST](https://medmnist.com/).
