# XAI Methodology

This repository implements Grad-CAM for CNN-style image classifiers under `src/xaimed/xai/gradcam.py` and exposes explanation generation via `xaimed explain`.

Current status:

- **Grad-CAM**: implemented and used to generate attribution overlays for sample images.
- **CLI explain workflow**: loads a trained checkpoint, runs Grad-CAM on a configured split, and saves per-sample overlays under `artifacts/explain/...`.
- **Integrated Gradients / sanity checks**: scaffolding remains for follow-up tasks.

The explain pipeline writes overlays named like `sample_000_t0_p1_gradcam.png` where `t` is target class and `p` is predicted class.

## Error Analysis Gallery

The evaluation stage now includes a confidence-based failure gallery selector:

- **High-confidence wrongs**: misclassifications sorted by confidence descending.
- **Low-confidence corrects**: correct predictions sorted by confidence ascending.
- Exports a ranked CSV and two image grids to `artifacts/eval/...`.

`failure_gallery_selection.csv` columns:

- `group`: Which selection bucket the sample belongs to (`high_confidence_wrong` or `low_confidence_correct`).
- `rank`: 1-based rank within the `group` after confidence sorting.
- `index`: Sample index in the evaluated dataset/dataloader order.
- `target`: Ground-truth class id.
- `prediction`: Predicted class id.
- `confidence`: Model confidence score (max softmax probability) used for ranking.

