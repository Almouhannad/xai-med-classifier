# Explainable Medical Classifier (XAI-first)

A high-performance classification pipeline for medical imaging using **MedMNIST**, designed with explainability as a first-class citizen.

## Quickstart

1. **Setup Environment**:
   ```bash
   make setup
   ```

2. **Run Smoke Test (Training)**:
   ```bash
   make train-smoke
   ```

3. **Generate Explanations**:
   ```bash
   make explain-smoke
   ```

4. **Generate Report**:
   ```bash
   make report
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
