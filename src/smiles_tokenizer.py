"""
Tokenization utilities for SMILES sequence generation.
"""

import re


PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


# Keep bracketed atoms like [O-] together, and keep two-letter atoms together.
# Everything else is treated as a single-character token.
TOKEN_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|.")


class SmilesTokenizer:
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        self.pad_idx = token_to_idx[PAD_TOKEN]
        self.start_idx = token_to_idx[START_TOKEN]
        self.end_idx = token_to_idx[END_TOKEN]
        self.unk_idx = token_to_idx[UNK_TOKEN]

    @classmethod
    def from_smiles_list(cls, smiles_list):
        # Build the vocabulary only from the dataset being used.
        unique_tokens = set()

        for smiles in smiles_list:
            unique_tokens.update(tokenize_smiles(smiles))

        ordered_tokens = SPECIAL_TOKENS + sorted(unique_tokens)
        token_to_idx = {token: idx for idx, token in enumerate(ordered_tokens)}
        return cls(token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, smiles, add_start=False, add_end=False):
        # The decoder sees START during training and learns to predict END.
        token_ids = []

        if add_start:
            token_ids.append(self.start_idx)

        for token in tokenize_smiles(smiles):
            token_ids.append(self.token_to_idx.get(token, self.unk_idx))

        if add_end:
            token_ids.append(self.end_idx)

        return token_ids

    def decode(self, token_ids, skip_special=True):
        # Stop decoding at END so padded values do not appear in predictions.
        tokens = []

        for token_id in token_ids:
            token = self.idx_to_token.get(int(token_id), UNK_TOKEN)

            if token == END_TOKEN:
                break

            if skip_special and token in SPECIAL_TOKENS:
                continue

            tokens.append(token)

        return "".join(tokens)


def tokenize_smiles(smiles):
    return TOKEN_PATTERN.findall(smiles)


def pad_sequence(token_ids, max_length, pad_idx):
    if len(token_ids) > max_length:
        return token_ids[:max_length]

    return token_ids + [pad_idx] * (max_length - len(token_ids))
