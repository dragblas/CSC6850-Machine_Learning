# OCSR Project Reference

This document explains the active project files and the purpose of each major
function/class. The current project is a CNN-GRU image-to-SMILES sequence model.

## Active Pipeline

```text
PubChem raw CSV
  -> RDKit image generation
  -> train/test split
  -> SMILES tokenization
  -> CNN encoder + GRU decoder training
  -> token, exact, canonical, and valid-SMILES evaluation
```

## Main Commands

Generate the current mixed PubChem dataset:

```bash
python3 -m scripts.generate_dataset --dataset pubchem
python3 -m scripts.split_pubchem --dataset pubchem
python3 -m scripts.validate_dataset --dataset pubchem
```

Train the model:

```bash
python3 -m src.train_sequence --dataset pubchem
```

Run a hyperparameter grid:

```bash
python3 -m scripts.run_hyperparameter_grid --dataset pubchem --epochs 10
```

Evaluate a trained checkpoint on an external dataset:

```bash
python3 -m scripts.evaluate_dataset --dataset epa --checkpoint sequence_baseline.pth
```

## Source Files

### `src/data.py`

General dataset utilities.

`get_default_transform()`

- Builds the default image preprocessing pipeline.
- Resizes molecule images to `224 x 224`.
- Converts images into PyTorch tensors.

`load_label_rows(label_path)`

- Reads a `labels.csv` file.
- Requires columns named `image` and `smiles`.
- Returns rows as dictionaries:

```python
{"image": "img_0001.png", "smiles": "CCO"}
```

`split_labels(label_path, train_path, test_path, train_ratio=0.8, seed=42)`

- Splits a labels CSV into train and test CSVs.
- Uses a fixed random seed for reproducibility.
- Writes `train_labels.csv` and `test_labels.csv`.

`MoleculeImageDataset`

- Basic dataset class for loading molecule images and SMILES strings.
- Returns:

```python
image, smiles
```

`validate_csv_schema(label_path)`

- Checks that a label CSV has the required columns.

`validate_images(image_dir, label_path)`

- Checks that every image listed in the CSV exists.

`validate_smiles(label_path)`

- Uses RDKit to check that each SMILES string can be parsed.

`validate_data(image_dir, label_path)`

- Runs schema, image, and SMILES validation together.

### `src/smiles_tokenizer.py`

SMILES tokenization utilities.

Special tokens:

- `<PAD>`: padding token for batching.
- `<START>`: first decoder input token.
- `<END>`: tells decoding to stop.
- `<UNK>`: fallback for unknown tokens.

`tokenize_smiles(smiles)`

- Splits a SMILES string into tokens.
- Keeps bracketed atoms like `[O-]` together.
- Keeps two-letter atoms like `Cl` and `Br` together.

`SmilesTokenizer.__init__(token_to_idx)`

- Stores token-to-index and index-to-token mappings.
- Stores special token indexes for easy access.

`SmilesTokenizer.from_smiles_list(smiles_list)`

- Builds a tokenizer vocabulary from a dataset.

`SmilesTokenizer.encode(smiles, add_start=False, add_end=False)`

- Converts a SMILES string into token IDs.
- Optionally adds `<START>` or `<END>`.

`SmilesTokenizer.decode(token_ids, skip_special=True)`

- Converts token IDs back into a SMILES string.
- Stops decoding when `<END>` appears.

`pad_sequence(token_ids, max_length, pad_idx)`

- Pads or truncates token sequences to a fixed length.

### `src/sequence_data.py`

Dataset utilities for sequence generation.

`SmilesSequenceDataset`

- Loads molecule images and creates decoder sequences.
- Returns:

```python
image, decoder_input, target
```

Example:

```text
SMILES:        CCO
decoder_input: <START> C C O
target:        C C O <END>
```

This setup is called teacher forcing.

`get_max_sequence_length(label_path, tokenizer)`

- Finds the longest tokenized SMILES string in a dataset.
- Adds one extra position for `<START>` or `<END>`.

### `src/seq_model.py`

CNN-GRU model definition.

`CNNEncoder`

- Convolutional image encoder.
- Converts a molecule image into a compact vector.
- Uses adaptive average pooling so the output feature size is stable.

`CNNEncoder.forward(images)`

- Applies the CNN to a batch of images.
- Returns an image embedding with shape:

```text
batch_size x hidden_dim
```

`ImageToSmilesModel`

- Full image-to-SMILES model.
- Components:
  - CNN encoder
  - token embedding layer
  - GRU decoder
  - linear output layer

`ImageToSmilesModel.forward(images, decoder_input)`

- Encodes the image.
- Uses the image embedding as the initial GRU hidden state.
- Feeds SMILES tokens into the GRU decoder.
- Returns token scores with shape:

```text
batch_size x sequence_length x vocab_size
```

### `src/evaluate.py`

Evaluation metrics.

`exact_match(predictions, targets)`

- Compares predicted and target SMILES as exact strings.

`canonicalize(smiles)`

- Uses RDKit to convert a SMILES string to canonical form.
- Returns `None` if the SMILES string is invalid.

`canonical_match(predictions, targets)`

- Compares predictions and targets after RDKit canonicalization.
- Gives credit for chemically equivalent SMILES strings.

`valid_smiles_rate(smiles_list)`

- Measures the fraction of generated SMILES strings that RDKit can parse.

`match(predictions, targets)`

- Backward-compatible alias for `exact_match`.

### `src/train_sequence.py`

Main training and evaluation script.

`train_sequence_model(...)`

- Main training function.
- Loads a dataset from `data/<dataset_name>`.
- Builds tokenizer and sequence datasets.
- Trains the CNN-GRU model.
- Evaluates token accuracy, exact match, canonical match, and valid SMILES rate.
- Saves metrics and model checkpoint.

Important command-line options:

```bash
--dataset
--batch-size
--epochs
--learning-rate
--hidden-dim
--embedding-dim
```

`evaluate_sequence_model(model, test_loader, tokenizer, device, max_length)`

- Evaluates model performance on the test set.
- Uses teacher-forced token accuracy and greedy decoded SMILES predictions.

`greedy_decode(model, images, tokenizer, device, max_length)`

- Generates SMILES strings one token at a time.
- Starts with `<START>`.
- Stops early if every generated sequence predicts `<END>`.

`save_metrics(metrics, output_path)`

- Appends evaluation results and hyperparameters to one shared CSV file.
- This keeps hyperparameter-grid runs in a single table.

## Script Files

### `scripts/generate_dataset.py`

Creates molecule images and `labels.csv` from a raw chemical CSV.

This script is used for PubChem and EPA. It can auto-detect common SMILES
columns such as `ConnectivitySMILES`, `SMILES`, `QSAR_READY_SMILES`, and
`MS_READY_SMILES`.

Examples:

```bash
python3 -m scripts.generate_dataset --dataset pubchem
python3 -m scripts.generate_dataset --dataset epa
```

Arguments:

```text
--dataset
--input
--output
--smiles-column
```

`generate_dataset(input_csv, smiles_column, output_dir)`

- Reads SMILES strings from a CSV.
- Uses RDKit to parse each molecule.
- Uses RDKit drawing tools to render each molecule image.
- Keeps the chemical name in `labels.csv` when a name column is available.
- Writes:

```text
data/<dataset>/images/
data/<dataset>/labels.csv
```

### `scripts/split_pubchem.py`

Creates train/test CSV files for a dataset.

Arguments:

```bash
--dataset
--train-ratio
--seed
```

Example:

```bash
python3 -m scripts.split_pubchem --dataset pubchem
```

### `scripts/validate_dataset.py`

Validates a dataset folder.

Example:

```bash
python3 -m scripts.validate_dataset --dataset pubchem
```

Arguments:

```text
--dataset
```

### `scripts/evaluate_dataset.py`

Evaluates a saved CNN-GRU checkpoint on an external dataset such as EPA.

Example:

```bash
python3 -m scripts.evaluate_dataset --dataset epa --checkpoint sequence_baseline.pth
```

Arguments:

```text
--dataset
--checkpoint
--batch-size
--output
```

This script does not train the model. It loads `sequence_baseline.pth`, runs
prediction on every row in `data/<dataset>/labels.csv`, and appends metrics to
`results/external_metrics.csv`.

### `scripts/run_hyperparameter_grid.py`

Runs repeated training experiments for hyperparameter tuning.

Default grid:

```text
learning rates: 0.001, 0.0005, 0.0001
hidden dims:    128, 256, 512
embedding dims: 128
```

Useful for the rubric requirement that multiple hyperparameter settings be
tested.

## Outputs

Training creates:

```text
sequence_baseline.pth
results/sequence_metrics.csv
```

`results/sequence_metrics.csv` is append-only. Each run adds one row containing
the model name, dataset, metrics, and hyperparameters.

Current result interpretation:

- The best current hyperparameter setting is learning rate `0.001`, hidden dim
  `512`, and embedding dim `128`.
- That run reached `0.7258` train token accuracy and `0.6932` test token
  accuracy.
- Exact match and canonical match are still `0.0`, which means token-level
  learning has not yet translated into full correct SMILES generation.
- Larger GRU hidden dimensions improved token accuracy in the current 27-run
  grid.
- Average test token accuracy increased from `0.5260` at hidden dim `128` to
  `0.6098` at hidden dim `512`.
- Average test token accuracy increased from `0.4849` at learning rate
  `0.0001` to `0.6367` at learning rate `0.001`.
- Valid SMILES rate only measures whether RDKit can parse the generated string;
  it does not mean the predicted molecule is correct.

The `.pth` file is a PyTorch checkpoint containing:

- model weights
- tokenizer vocabulary
- max sequence length
- model hyperparameters

The `.pth` file should usually not be committed to GitHub.
