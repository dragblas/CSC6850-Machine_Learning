# Dataset Format

All datasets should use the same basic layout:

```text
data/<dataset_name>/
  images/
  labels.csv
```

`labels.csv` must contain:

```csv
image,smiles
img_0001.png,CCO
img_0002.png,c1ccccc1
```

This shared format supports the image-to-SMILES sequence model:

- CNN-GRU image-to-SMILES sequence model

Current dataset plan:

- `data/pubchem`: mixed PubChem subset currently used for baseline experiments
- `data/epa`: small EPA CompTox subset intended for external testing

The project previously considered a separate `pubchem_simple` subset, but that
optional path was removed to keep the active codebase focused on one runnable
PubChem workflow.

Common dataset commands:

```bash
python3 -m scripts.generate_dataset --dataset pubchem
python3 -m scripts.generate_dataset --dataset epa
python3 -m scripts.validate_dataset --dataset pubchem
python3 -m scripts.validate_dataset --dataset epa
python3 -m scripts.evaluate_dataset --dataset epa --checkpoint sequence_baseline.pth
```

Only split datasets that will be used for training. PubChem is split because it
is the current training dataset. EPA is intended as an external test dataset, so
it does not need a train/test split. Instead, use `scripts.evaluate_dataset` to
test a PubChem-trained checkpoint on EPA.
