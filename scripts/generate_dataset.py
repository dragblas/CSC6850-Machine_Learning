"""
Generate molecule images and labels from a raw chemical CSV.

This script is used for PubChem and can also be used for EPA CompTox exports as
long as the input CSV contains a SMILES column.
"""

import argparse
import csv
import os


SMILES_COLUMN_CANDIDATES = [
    "ConnectivitySMILES",
    "SMILES",
    "QSAR_READY_SMILES",
    "QSAR-Ready SMILES",
    "MS_READY_SMILES",
    "MS-Ready SMILES",
]

NAME_COLUMN_CANDIDATES = [
    "Chemical Name",
    "PREFERRED_NAME",
    "Preferred Name",
    "INPUT",
]


DATASET_DEFAULTS = {
    "pubchem": {
        "input": "data/pubchem/raw.csv",
        "output": "data/pubchem",
        "smiles_column": "ConnectivitySMILES",
    },
    "epa": {
        "input": "data/epa/raw.csv",
        "output": "data/epa",
        "smiles_column": "SMILES",
    },
}


def resolve_column(fieldnames, requested_column, candidates):
    """Find the requested column or fall back to common column names."""
    if requested_column in fieldnames:
        return requested_column

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    raise ValueError(
        "Could not find a usable column. "
        f"Requested: {requested_column}. Available columns: {fieldnames}"
    )


def get_dataset_paths(dataset_name, input_csv=None, output_dir=None, smiles_column=None):
    """Resolve CLI options into input/output paths for a named dataset."""
    defaults = DATASET_DEFAULTS.get(dataset_name, {})

    return {
        "input_csv": input_csv or defaults.get("input", f"data/{dataset_name}/raw.csv"),
        "output_dir": output_dir or defaults.get("output", f"data/{dataset_name}"),
        "smiles_column": smiles_column or defaults.get("smiles_column", "SMILES"),
    }


def generate_dataset(input_csv, smiles_column, output_dir):
    """Render molecule images from a CSV column containing SMILES strings."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    # The model expects each dataset to have an images/ folder and labels.csv.
    images_dir = os.path.join(output_dir, "images")
    labels_path = os.path.join(output_dir, "labels.csv")

    # Re-running this script can reuse the folder and overwrite matching files.
    os.makedirs(images_dir, exist_ok=True)

    # These counters show how much of the raw CSV was usable.
    count_total = 0
    count_valid = 0
    count_invalid = 0

    with open(input_csv, "r", encoding="utf-8") as infile, open(
        labels_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")

        smiles_column = resolve_column(
            reader.fieldnames,
            smiles_column,
            SMILES_COLUMN_CANDIDATES,
        )
        name_column = None
        for candidate in NAME_COLUMN_CANDIDATES:
            if candidate in reader.fieldnames:
                name_column = candidate
                break

        writer = csv.writer(outfile)
        if name_column:
            writer.writerow(["image", "smiles", "name"])
        else:
            writer.writerow(["image", "smiles"])

        for row in reader:
            count_total += 1
            smiles = row.get(smiles_column, "").strip()

            if not smiles:
                count_invalid += 1
                continue

            # RDKit returns None when the SMILES cannot be parsed as a molecule.
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                count_invalid += 1
                continue

            # Number images in the order they are successfully generated.
            image_name = f"img_{count_valid:04d}.png"
            image_path = os.path.join(images_dir, image_name)

            # RDKit draws a standard 2D structure image for the molecule.
            img = Draw.MolToImage(mol)
            img.save(image_path)

            # labels.csv is the connection between each image and its SMILES.
            if name_column:
                writer.writerow([image_name, smiles, row.get(name_column, "").strip()])
            else:
                writer.writerow([image_name, smiles])
            count_valid += 1

    print("Dataset generation complete.")
    print("Input:", input_csv)
    print("Output:", output_dir)
    print("SMILES column:", smiles_column)
    print("Total:", count_total)
    print("Valid:", count_valid)
    print("Invalid:", count_invalid)


if __name__ == "__main__":
    # Command-line arguments let the same script generate any dataset folder
    # that follows the expected labels.csv/images layout.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubchem")
    parser.add_argument("--input", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = get_dataset_paths(
        dataset_name=args.dataset,
        input_csv=args.input,
        output_dir=args.output,
        smiles_column=args.smiles_column,
    )

    generate_dataset(
        input_csv=paths["input_csv"],
        smiles_column=paths["smiles_column"],
        output_dir=paths["output_dir"],
    )
