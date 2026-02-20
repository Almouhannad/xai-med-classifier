# XAI Methodology (Scaffold)

This repository reserves XAI modules for Grad-CAM, Integrated Gradients, and sanity checks under `src/xaimed/xai/`.

Current status:

- Public APIs exist and are intentionally scaffolded.
- CLI wiring (`xaimed explain`) is in place and returns a scaffold message.
- Future tasks will implement attribution generation and artifact persistence.

This scaffold keeps interfaces stable while incremental work implement each method with tests.
