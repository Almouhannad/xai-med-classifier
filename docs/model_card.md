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

The factory supports configurable input channels. For non-RGB data (`in_channels != 3`), the first convolution is replaced and weights are remapped from the original RGB kernels (mean for 1 channel, slice for <3 channels, repeat for >3 channels). This preserves pretrained conv1 information when `pretrained=True`.

## Verification

Unit tests validate output tensor shapes for dummy batches and verify conv1 remapping behavior during input-channel adaptation.