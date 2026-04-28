# DECIMER

Use DECIMER as an external hand-drawn test set.

1. Download from:

```text
https://zenodo.org/records/7617107
```

2. Download:

```text
DECIMER_HDM_Dataset_Images.zip
DECIMER_HDM_Dataset_SMILES.tsv
```

3. Extract/copy the PNG files into:

```text
data/decimer/images/
```

4. Save the TSV as:

```text
data/decimer/DECIMER_HDM_Dataset_SMILES.tsv
```

5. Convert the TSV to `labels.csv` and validate:

```bash
python3 -m scripts.generate_dataset --dataset decimer
python3 -m scripts.validate_dataset --dataset decimer
```

6. Evaluate:

```bash
python3 -m scripts.evaluate_dataset --dataset decimer --limit 100
```
