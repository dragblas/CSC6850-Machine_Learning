# PubChem

Use PubChem as the training dataset.

Run from the repository root:

```bash
mkdir -p data/pubchem
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(seq -s, 1000 1399)/property/ConnectivitySMILES/CSV" -o data/pubchem/raw.csv
python3 -m scripts.generate_dataset --dataset pubchem --smiles-column ConnectivitySMILES
python3 -m scripts.split_pubchem --dataset pubchem
python3 -m scripts.validate_dataset --dataset pubchem
```

Then train:

```bash
python3 -m src.train_sequence --dataset pubchem --decoder-type gru
```
