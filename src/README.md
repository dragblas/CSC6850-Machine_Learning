# Source

Main files:

```text
data.py              load, split, and validate datasets
evaluate.py          SMILES metrics
smiles_tokenizer.py  SMILES tokenization
sequence_data.py     image-to-SMILES dataset wrapper
seq_model.py         CNN encoder + RNN/GRU/LSTM decoder
train_sequence.py    training entry point
```

Train:

```bash
python3 -m src.train_sequence --dataset pubchem --decoder-type gru
```

Common options:

```text
--dataset
--batch-size
--epochs
--learning-rate
--hidden-dim
--embedding-dim
--decoder-type   rnn, gru, or lstm
--save-path
```
