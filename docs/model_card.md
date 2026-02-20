# Model Card (Current Baseline)

## Architecture

The project uses a model factory (`src/xaimed/models/factory.py`) to instantiate torchvision ResNet backbones for classification.

Supported backbones:
- `resnet18`
- `resnet34`
- `resnet50`
- `resnet101`

## Head Adaptation

For each backbone, the final fully connected classifier is replaced so the output dimension exactly matches the configured `num_classes`.

## Input Channels

The factory supports configurable input channels. For non-RGB data (`in_channels != 3`), the first convolution is replaced and weights are remapped from the original RGB kernels (mean for 1 channel; repeat/truncate and scale by `3 / in_channels` otherwise). This preserves pretrained conv1 information when `pretrained=True` and mitigates activation-scale drift when `in_channels > 3`.


## Weights & Preprocessing

The model factory supports `pretrained=True` (mapped to torchvision `DEFAULT` weights) or an explicit `weights` value. Use only one of these options at a time.

When pretrained weights are used, inputs should be preprocessed with the corresponding torchvision transforms (`weights.transforms()`) to match the expected normalization and resize/crop behavior.

## Verification

Unit tests validate output tensor shapes for dummy batches and verify conv1 remapping behavior during input-channel adaptation.