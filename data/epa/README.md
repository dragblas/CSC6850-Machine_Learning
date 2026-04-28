# EPA Dataset Notes

This folder is for the small EPA CompTox dataset.

## Download Source

Use the EPA CompTox Batch Search page:

```text
https://comptox.epa.gov/dashboard/batch-search
```

## Manual Download Steps

1. Open the EPA CompTox Batch Search page.
2. Copy the chemical names from:

```text
data/epa/chemical_names.txt
```

3. Paste those names into the batch search input.
4. In the export/data options, select:

```text
Chemical Name
SMILES
```

5. Download the result as a CSV file.
6. Save the downloaded file as:

```text
data/epa/raw.csv
```

## Generate EPA Images

After `data/epa/raw.csv` exists, generate molecule images and `labels.csv`:

```bash
python3 -m scripts.generate_dataset --dataset epa
```

The generator automatically recognizes common SMILES columns such as `SMILES`,
`ConnectivitySMILES`, `QSAR_READY_SMILES`, and `MS_READY_SMILES`.

If needed, you can still specify the column manually:

```bash
python3 -m scripts.generate_dataset --dataset epa --smiles-column SMILES
```

## Validate

Validate the dataset:

```bash
python3 -m scripts.validate_dataset --dataset epa
```

## Expected Format

After generation, this folder should look like:

```text
data/epa/
  raw.csv
  chemical_names.txt
  images/
  labels.csv
```

EPA is intended as an external test dataset, so it does not need a train/test
split unless you choose to train a separate EPA-only model later.

## External Evaluation

After the PubChem-trained checkpoint exists, evaluate it on EPA:

```bash
python3 -m scripts.evaluate_dataset --dataset epa --checkpoint sequence_baseline.pth
```

This writes external-test metrics to:

```text
results/external_metrics.csv
```
