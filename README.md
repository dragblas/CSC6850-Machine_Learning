# CSC6850 Machine Learning

```text
chemical image -> CNN encoder -> RNN/GRU/LSTM decoder -> SMILES tokens
```

## Setup

Run commands from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## PubChem Training Data

Download, generate, split, and validate PubChem:

```bash
mkdir -p data/pubchem
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(seq -s, 1000 1399)/property/ConnectivitySMILES/CSV" -o data/pubchem/raw.csv
python3 -m scripts.generate_dataset --dataset pubchem --smiles-column ConnectivitySMILES
python3 -m scripts.split_pubchem --dataset pubchem
python3 -m scripts.validate_dataset --dataset pubchem
```

## Train

Train one decoder:

```bash
python3 -m src.train_sequence --dataset pubchem --decoder-type gru
```

Train all decoder types with the grid script:

```bash
python3 -m scripts.run_hyperparameter_grid --dataset pubchem --decoder-type rnn,gru,lstm
```

Checkpoints save as:

```text
sequence_rnn.pth
sequence_gru.pth
sequence_lstm.pth
```

Training metrics append to:

```text
results/sequence_metrics.csv
```

## External Evaluation

After you train on PubChem, evaluate EPA or DECIMER:

```bash
python3 -m scripts.evaluate_dataset --dataset epa
python3 -m scripts.evaluate_dataset --dataset decimer
```

External metrics append to:

```text
results/external_metrics.csv
```

## Useful Flags

```text
src.train_sequence
  --dataset         dataset folder under data/
  --batch-size      samples per training batch
  --epochs          full passes through the training set
  --learning-rate   optimizer step size
  --hidden-dim      recurrent decoder hidden size
  --embedding-dim   SMILES token embedding size
  --decoder-type    rnn, gru, or lstm
  --save-path       optional checkpoint filename

scripts.run_hyperparameter_grid
  --dataset         dataset folder under data/
  --learning-rates  comma-separated learning rates
  --hidden-dims     comma-separated hidden sizes
  --embedding-dims  comma-separated embedding sizes
  --decoder-type    one decoder type or comma-separated decoder types
  --epochs          epochs per run
  --batch-size      samples per batch

scripts.generate_dataset
  --dataset        pubchem, epa, or decimer
  --input          optional raw CSV/TSV path
  --output         optional dataset output folder
  --smiles-column  optional override; otherwise uses SMILES or smiles

scripts.evaluate_dataset
  --dataset        epa or decimer
  --decoder-type   rnn, gru, or lstm; default is gru
  --checkpoint     optional checkpoint override
  --batch-size     samples per evaluation batch
  --limit          optional number of rows to evaluate
  --output         optional metrics CSV path
```

See the dataset READMEs under `data/` for EPA and DECIMER download steps.
