# Test Folder

This folder has scripts for testing, debugging, and preparing datasets.
They are not CORE model implementation but are used to verify data pipeline (before training).

## Data Generation

### `generate_test_data.py`
Creates a small synthetic dataset using RDKit.
- Generates simple molecule images
- Creates `data/test/images/` and `data/test/labels.csv`
- For early debugging\

Usage:
```bash
python -m test.generate_test_data.py
```

### `generate_dataset.py`
Data set generator from SMILES
- Input: CSV file containing SMILE Strings
- Ouput: `images/` and `labels.csv`
- core function for dataset creation

Usage:
```bash
python -m test.generate_dataset.py
```

### `generate_pubchem.py`
Build PubChem Dataset
- Input `data/pubchem/raw.csv`
- Uses SMILES column from PubChem
- Output: `data/pubchem/images/` and `data/pubchem/labels.csv`

Usage:
```bash
python -m test.generate_pubchem.py
```

## Validation
### `test_validate_data.py`
Runs full dataset validation
Checks:
- CSV Schema (`image`, `smiles`)
- image file existence
- SMILES validity (RDKit)

Usage:
```bash
python -m test.test_validate_data <dataset_name>
```

### `validate_smiles.py`
Debug tool for SMILES 
- iterates through dataset 
- counts valid vs invalid SMILES

Usage:
```bash
python -m test.validate_smiles <dataset_name>
```

## Pipeline Testing
### `test_load_data.py`
Tests dataset loading pipeline
- Loads dataset using Dataset class
- Applies transofmrations (resize, tensor)
Verifies
- image loading
- SMILES pairing
- tensor shape

Usage:
```bash
python -m test.test_load_data <dataset_name>
```

### `test_comparisons.py`
Tests evaluation metrics
- tests exact match(`match()`)
- tests chemical equivalence (`canonical_match()`)
Used to verify correctness of logic

Usage:
```bash
python -m test.test_comparisons
```