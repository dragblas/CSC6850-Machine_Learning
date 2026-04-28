# CSC6850 Machine Learning

Optical Chemical Structure Recognition baseline project.

This repo currently focuses on a CNN-GRU image-to-SMILES sequence baseline:

```text
chemical image -> CNN encoder -> GRU decoder -> SMILES tokens
```

This model is closer to the OCSR formulation used by SwinOCSR because it
generates a molecular string token-by-token rather than choosing a molecule ID.

## Structure

```text
src/
  data.py       dataset loading, transforms, splitting, validation
  evaluate.py   exact and canonical SMILES metrics
  smiles_tokenizer.py
  sequence_data.py
  seq_model.py
  train_sequence.py

scripts/
  generate_dataset.py
  evaluate_dataset.py
  run_hyperparameter_grid.py
  split_pubchem.py
  validate_dataset.py
```

## How to Run

Run all commands from the project root:

```bash
cd /Users/dragblas/Coding/CSC6850-Machine_Learning
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download the PubChem raw CSV:

```bash
mkdir -p data/pubchem
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(seq -s, 1000 1399)/property/ConnectivitySMILES/CSV" -o data/pubchem/raw.csv
```

Generate the PubChem image dataset from `data/pubchem/raw.csv`:

```bash
python3 -m scripts.generate_dataset --dataset pubchem
```

Create the 80/20 train/test split:

```bash
python3 -m scripts.split_pubchem --dataset pubchem
```

Validate the dataset:

```bash
python3 -m scripts.validate_dataset --dataset pubchem
```

Script flag summary:

```text
scripts.generate_dataset
  --dataset         Dataset name. Supported defaults: pubchem, epa
  --input           Optional raw CSV override
  --output          Optional output folder override
  --smiles-column   Optional SMILES column override

scripts.split_pubchem
  --dataset         Dataset folder under data/. Default: pubchem
  --train-ratio     Fraction of labels used for training. Default: 0.8
  --seed            Random seed for reproducible splits. Default: 42

scripts.validate_dataset
  --dataset         Dataset folder under data/. Default: pubchem

scripts.evaluate_dataset
  --dataset         External dataset folder under data/. Default: epa
  --checkpoint      Saved model checkpoint. Default: sequence_baseline.pth
  --batch-size      Number of samples per evaluation batch. Default: 16
  --output          Metrics CSV path. Default: results/external_metrics.csv
```

Full from-scratch workflow:

```bash
mkdir -p data/pubchem
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(seq -s, 1000 1399)/property/ConnectivitySMILES/CSV" -o data/pubchem/raw.csv
python3 -m scripts.generate_dataset --dataset pubchem
python3 -m scripts.split_pubchem --dataset pubchem
python3 -m scripts.validate_dataset --dataset pubchem
python3 -m src.train_sequence
```

Train and evaluate the CNN-GRU image-to-SMILES sequence baseline:

```bash
python3 -m src.train_sequence
```

Train on a specific dataset or hyperparameter setting:

```bash
python3 -m src.train_sequence --dataset pubchem --epochs 10 --learning-rate 0.001 --hidden-dim 256 --embedding-dim 128
```

`src.train_sequence` flags:

```text
--dataset         Dataset folder under data/ to use. Default: pubchem
--batch-size      Number of samples per batch before a weight update. Default: 16
--epochs          Number of full passes over the training set. Default: 10
--learning-rate   Optimizer step size for weight updates. Default: 0.001
--hidden-dim      GRU hidden-state size. Default: 256
--embedding-dim   SMILES token embedding size. Default: 128
```

Run a small hyperparameter grid:

```bash
python3 -m scripts.run_hyperparameter_grid --dataset pubchem --epochs 10
```

Evaluate a trained PubChem checkpoint on EPA as an external dataset:

```bash
python3 -m scripts.evaluate_dataset --dataset epa --checkpoint sequence_baseline.pth
```

Sequence training saves:

```text
sequence_baseline.pth
results/sequence_metrics.csv
results/external_metrics.csv
```

`results/sequence_metrics.csv` is append-only. Each training run adds one row
with the dataset, metrics, and hyperparameters so grid-search results stay in
one table.
