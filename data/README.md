# Data

Use this layout for every dataset:

```text
data/<dataset>/
  images/
  labels.csv
```

`labels.csv` must contain:

```csv
image,smiles
img_0001.png,CCO
```

Use PubChem for training. Use EPA and DECIMER as external test sets.

Do not commit generated images, raw downloads, labels, splits, checkpoints, or
results. They are ignored in `.gitignore`.
