# EPA

Use EPA as an external test set.

1. Open:

```text
https://comptox.epa.gov/dashboard/batch-search
```

2. Paste the names from:

```text
data/epa/chemical_names.txt
```

3. Export:

```text
Chemical Name
SMILES
```

4. Save the download as:

```text
data/epa/raw.csv
```

5. Generate and validate:

```bash
python3 -m scripts.generate_dataset --dataset epa
python3 -m scripts.validate_dataset --dataset epa
```

6. Evaluate:

```bash
python3 -m scripts.evaluate_dataset --dataset epa
```
