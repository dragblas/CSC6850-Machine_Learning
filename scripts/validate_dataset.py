"""
Validate a dataset folder with images/ and labels.csv.
"""

import argparse
import os

from src.data import validate_data


def main():
    # The dataset argument is the folder name under data/.
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_arg", nargs="?", help="Dataset name under data/")
    parser.add_argument("--dataset", dest="dataset_flag", help="Dataset name under data/")
    args = parser.parse_args()
    dataset = args.dataset_flag or args.dataset_arg or "pubchem"

    # validate_data expects the standard dataset layout.
    image_dir = os.path.join("data", dataset, "images")
    label_path = os.path.join("data", dataset, "labels.csv")

    # The validation function checks CSV format, image files, and SMILES parsing.
    overall_ok, results = validate_data(image_dir, label_path)

    print("Overall Valid:", overall_ok)

    # Print each validation category separately so failures are easy to locate.
    for category, (status, issues) in results.items():
        print(f"\n{category.upper()} VALID:", status)
        for issue in issues:
            print(issue)


if __name__ == "__main__":
    main()
