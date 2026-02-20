# XAI Methodology (Scaffold)

This repository reserves XAI modules for Grad-CAM, Integrated Gradients, and sanity checks under `src/xaimed/xai/`.

Current status:

- Public APIs exist and are intentionally scaffolded.
- CLI wiring (`xaimed explain`) is in place and returns a scaffold message.
- Future tasks will implement attribution generation and artifact persistence.

This scaffold keeps interfaces stable while incremental work implement each method with tests.

## Error Analysis Gallery

The evaluation stage now includes a confidence-based failure gallery selector:

- **High-confidence wrongs**: misclassifications sorted by confidence descending.
- **Low-confidence corrects**: correct predictions sorted by confidence ascending.
- Exports a ranked CSV and two image grids to `artifacts/eval/...`.
