# Source Files

```text
data.py              dataset loading, transforms, split, validation
evaluate.py          exact and canonical SMILES metrics
smiles_tokenizer.py  SMILES token vocabulary and encode/decode helpers
sequence_data.py     image-to-SMILES sequence dataset
seq_model.py         CNN encoder + GRU decoder
train_sequence.py    sequence model training/evaluation entry point
```

Run the sequence baseline:

```bash
python3 -m src.train_sequence
```

Useful options:

```bash
python3 -m src.train_sequence --dataset pubchem --epochs 10 --batch-size 16 --learning-rate 0.001 --hidden-dim 256 --embedding-dim 128
```

Flag summary:

```text
--dataset         Dataset folder under data/ to use. Default: pubchem
--batch-size      Number of samples per batch before a weight update. Default: 16
--epochs          Number of full passes over the training set. Default: 10
--learning-rate   Optimizer step size for weight updates. Default: 0.001
--hidden-dim      GRU hidden-state size. Default: 256
--embedding-dim   SMILES token embedding size. Default: 128
```
