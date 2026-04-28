"""
Create the default PubChem train/test split.
"""

import argparse
import os

from src.data import split_labels


if __name__ == "__main__":
    # This script is a small command-line wrapper around src.data.split_labels.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubchem")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # All datasets live under data/<dataset name>/.
    dataset_dir = os.path.join("data", args.dataset)

    # Creates train_labels.csv and test_labels.csv from the full labels.csv.
    split_labels(
        label_path=os.path.join(dataset_dir, "labels.csv"),
        train_path=os.path.join(dataset_dir, "train_labels.csv"),
        test_path=os.path.join(dataset_dir, "test_labels.csv"),
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
