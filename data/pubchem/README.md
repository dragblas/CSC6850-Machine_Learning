## Reproducing Dataset

Run all commands from the project root.

1. Download PubChem subset:

You can change the CID range in the curl command to change the dataset size.

```bash
curl "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$(seq -s, 1000 1399)/property/ConnectivitySMILES/CSV" -o data/pubchem/raw.csv
```

2. Generate images + labels:

```bash
python3 -m scripts.generate_dataset --dataset pubchem
```

3. Create train/test split:

```bash
python3 -m scripts.split_pubchem --dataset pubchem
```

4. Validate:

```bash
python3 -m scripts.validate_dataset --dataset pubchem
```

## Current Role

This folder contains the mixed PubChem subset currently used for baseline
experiments.
