"""
Generate labels and/or molecule images from raw dataset files.
"""

import argparse
import csv
import os


DATASET_DEFAULTS = {
    "pubchem": {
        "input": "data/pubchem/raw.csv",
        "output": "data/pubchem",
        "existing_images": False,
    },
    "epa": {
        "input": "data/epa/raw.csv",
        "output": "data/epa",
        "existing_images": False,
    },
    "decimer": {
        "input": "data/decimer/DECIMER_HDM_Dataset_SMILES.tsv",
        "output": "data/decimer",
        "existing_images": True,
    },
}


def get_dataset_paths(dataset_name, input_path=None, output_dir=None):
    """Resolve the input file and output folder for a dataset."""
    defaults = DATASET_DEFAULTS.get(dataset_name, {})

    return {
        "input_path": input_path or defaults.get("input", f"data/{dataset_name}/raw.csv"),
        "output_dir": output_dir or defaults.get("output", f"data/{dataset_name}"),
        "existing_images": defaults.get("existing_images", False),
    }


def get_delimiter(input_path):
    """Use tab-separated parsing for TSV files."""
    return "\t" if input_path.lower().endswith(".tsv") else ","


def resolve_smiles_column(fieldnames, requested_column=None):
    """Use an explicit column, or default to SMILES/smiles."""
    if requested_column:
        if requested_column in fieldnames:
            return requested_column

        raise ValueError(
            f"Could not find requested SMILES column: {requested_column}. "
            f"Available columns: {fieldnames}"
        )

    if "SMILES" in fieldnames:
        return "SMILES"

    if "smiles" in fieldnames:
        return "smiles"

    raise ValueError(
        "Could not find a SMILES column. Expected 'SMILES' or 'smiles'. "
        f"Available columns: {fieldnames}"
    )


def resolve_image_column(fieldnames, smiles_column):
    """Use the first non-SMILES column as the image id/filename column."""
    for fieldname in fieldnames:
        if fieldname != smiles_column:
            return fieldname

    raise ValueError("Could not find an image identifier column.")


def get_value(row, column):
    """Read and strip one CSV/TSV cell."""
    value = row.get(column, "")
    return "" if value is None else value.strip()


def generate_dataset(input_path, smiles_column, output_dir):
    """Draw molecule images from a CSV/TSV file containing SMILES strings."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    images_dir = os.path.join(output_dir, "images")
    labels_path = os.path.join(output_dir, "labels.csv")
    os.makedirs(images_dir, exist_ok=True)

    count_total = 0
    count_valid = 0
    count_invalid = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        labels_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile, delimiter=get_delimiter(input_path))
        if reader.fieldnames is None:
            raise ValueError("Input file is missing a header row.")

        smiles_column = resolve_smiles_column(reader.fieldnames, smiles_column)

        writer = csv.writer(outfile)
        writer.writerow(["image", "smiles"])

        for row in reader:
            count_total += 1
            smiles = get_value(row, smiles_column)

            if not smiles:
                count_invalid += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                count_invalid += 1
                continue

            image_name = f"img_{count_valid:04d}.png"
            image_path = os.path.join(images_dir, image_name)

            Draw.MolToImage(mol).save(image_path)
            writer.writerow([image_name, smiles])
            count_valid += 1

    print("Dataset generation complete.")
    print("Input:", input_path)
    print("Output:", output_dir)
    print("SMILES column:", smiles_column)
    print("Total:", count_total)
    print("Valid:", count_valid)
    print("Invalid:", count_invalid)


def generate_labels_for_existing_images(input_path, smiles_column, output_dir):
    """Convert a CSV/TSV file into labels.csv for existing images."""
    from rdkit import Chem

    images_dir = os.path.join(output_dir, "images")
    labels_path = os.path.join(output_dir, "labels.csv")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Expected image folder not found: {images_dir}"
        )

    image_lookup = {
        os.path.splitext(filename)[0]: filename
        for filename in os.listdir(images_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    }

    count_total = 0
    count_written = 0
    count_missing_image = 0
    count_missing_smiles = 0
    count_invalid_smiles = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(
        labels_path, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile, delimiter=get_delimiter(input_path))
        if reader.fieldnames is None:
            raise ValueError("Input file is missing a header row.")

        smiles_column = resolve_smiles_column(reader.fieldnames, smiles_column)
        image_column = resolve_image_column(reader.fieldnames, smiles_column)

        writer = csv.writer(outfile)
        writer.writerow(["image", "smiles"])

        for row in reader:
            count_total += 1
            smiles = get_value(row, smiles_column)
            image_value = get_value(row, image_column)

            if not smiles:
                count_missing_smiles += 1
                continue

            # Keep labels.csv compatible with the validator/evaluator by
            # dropping rows RDKit cannot parse.
            if Chem.MolFromSmiles(smiles) is None:
                count_invalid_smiles += 1
                continue

            image_name = match_existing_image(image_value, image_lookup)
            if image_name is None:
                count_missing_image += 1
                continue

            writer.writerow([image_name, smiles])
            count_written += 1

    print("Existing-image label generation complete.")
    print("Input:", input_path)
    print("Output:", labels_path)
    print("Image column:", image_column)
    print("SMILES column:", smiles_column)
    print("Total:", count_total)
    print("Written:", count_written)
    print("Missing image:", count_missing_image)
    print("Missing SMILES:", count_missing_smiles)
    print("Invalid SMILES:", count_invalid_smiles)


def match_existing_image(image_value, image_lookup):
    """Match an image id or filename to a file in images/."""
    if not image_value:
        return None

    image_filename = os.path.basename(image_value)
    image_stem, image_ext = os.path.splitext(image_filename)

    if image_ext:
        return image_lookup.get(image_stem)

    return image_lookup.get(image_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pubchem")
    parser.add_argument("--input", default=None)
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = get_dataset_paths(
        dataset_name=args.dataset,
        input_path=args.input,
        output_dir=args.output,
    )

    if paths["existing_images"]:
        generate_labels_for_existing_images(
            input_path=paths["input_path"],
            smiles_column=args.smiles_column,
            output_dir=paths["output_dir"],
        )
    else:
        generate_dataset(
            input_path=paths["input_path"],
            smiles_column=args.smiles_column,
            output_dir=paths["output_dir"],
        )
