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

The factory supports configurable input channels. For non-RGB data (`in_channels != 3`), the first convolution is replaced to accept the requested number of channels.

## Verification

Unit tests validate output tensor shapes for dummy batches for supported ResNet variants.