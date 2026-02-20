# Dataset Card: MedMNIST Integration

## Source
- Dataset family: [MedMNIST](https://medmnist.com/)
- Primary example in this repository: `pathmnist`

## Local Storage
- Download location is configurable with `--data-dir`.
- Default location in helper script: `data/`.
- Generated files should not be committed.

## Download Commands

Using helper script:

```bash
python scripts/download_data.py --dataset pathmnist --data-dir data
```

Using package CLI:

```bash
python -m xaimed.cli download-data --dataset pathmnist --data-dir data
```

## Data Loading API

Implemented in `src/xaimed/data/medmnist.py`:
- `build_transforms(image_size=224)`
- `load_medmnist_dataset(dataset_name, split, data_dir, download, transform)`
- `build_medmnist_dataloaders(dataset_name, data_dir, batch_size, num_workers, image_size)`
- `download_medmnist(dataset_name, data_dir)`

## Validation
- Download smoke check should complete and print split counts.
- Dataloader smoke check should construct train/val/test loaders and iterate one batch.