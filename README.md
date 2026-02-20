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

4. **Generate Explanations**:
   ```bash
   make explain-smoke
   ```

5. **Generate Report**:
   ```bash
   make report
   ```

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
