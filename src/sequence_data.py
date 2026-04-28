"""
Dataset utilities for image-to-SMILES sequence generation.
"""

import os

import torch
from PIL import Image

from src.data import load_label_rows
from src.smiles_tokenizer import pad_sequence


class SmilesSequenceDataset:
    def __init__(self, image_dir, label_path, tokenizer, max_length, transform=None):
        self.image_dir = image_dir
        self.samples = load_label_rows(label_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Each sample returns one image and two shifted SMILES sequences.
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # decoder_input: <START> C C O
        # target:        C C O <END>
        decoder_input = self.tokenizer.encode(sample["smiles"], add_start=True)
        target = self.tokenizer.encode(sample["smiles"], add_end=True)

        # Pad all sequences to the same length for batching
        decoder_input = pad_sequence(
            decoder_input,
            max_length=self.max_length,
            pad_idx=self.tokenizer.pad_idx,
        )
        target = pad_sequence(
            target,
            max_length=self.max_length,
            pad_idx=self.tokenizer.pad_idx,
        )

        return (
            image,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def get_max_sequence_length(label_path, tokenizer):
    # calculates max sequenc length (for padding)

    # Add one position for START or END.
    rows = load_label_rows(label_path)
    max_smiles_length = max(
        len(tokenizer.encode(row["smiles"]))
        for row in rows
    )
    return max_smiles_length + 1
