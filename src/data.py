"""
Data utilities for the image-to-SMILES OCSR baseline.
"""

import csv
import os
import random



def get_default_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_label_rows(label_path):
    """
    function to load the labels.csv file and return a list of dictionaries with keys "image" and "smiles"
    """

    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV file is missing a header row.")

        required_columns = {"image", "smiles"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        return [
            {"image": row["image"].strip(), "smiles": row["smiles"].strip()}
            for row in reader
        ]


def split_labels(label_path, train_path, test_path, train_ratio=0.8, seed=42):
    """ 
    splits the labels.csv file into train and test sets based on the specified ratio and random seed
    """
    random.seed(seed)

    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # shuffles rows, and splits based on index (train_ratio)
    random.shuffle(rows)
    split_index = int(len(rows) * train_ratio)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(train_rows)

    with open(test_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(test_rows)

    print("Split complete.")
    print("Train samples:", len(train_rows))
    print("Test samples:", len(test_rows))

    return train_rows, test_rows


class MoleculeImageDataset:
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = load_label_rows(label_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample["image"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, sample["smiles"]


def validate_csv_schema(label_path):
    issues = []

    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            issues.append("CSV file is missing a header row.")
            return False, issues

        required_columns = {"image", "smiles"}
        missing = required_columns - set(reader.fieldnames)
        if missing:
            issues.append(f"CSV missing required columns: {missing}")

    return len(issues) == 0, issues


def validate_images(image_dir, label_path):
    issues = []

    for i, row in enumerate(load_label_rows(label_path), start=1):
        image_path = os.path.join(image_dir, row["image"])
        if not os.path.exists(image_path):
            issues.append(f"Row {i}: missing image file -> {image_path}")

    return len(issues) == 0, issues


def validate_smiles(label_path):
    from rdkit import Chem

    issues = []

    for i, row in enumerate(load_label_rows(label_path), start=1):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            issues.append(f"Row {i}: invalid SMILES string -> {row['smiles']}")

    return len(issues) == 0, issues


def validate_data(image_dir, label_path):
    schema_ok, schema_issues = validate_csv_schema(label_path)
    images_ok, image_issues = validate_images(image_dir, label_path)
    smiles_ok, smiles_issues = validate_smiles(label_path)

    results = {
        "schema": (schema_ok, schema_issues),
        "images": (images_ok, image_issues),
        "smiles": (smiles_ok, smiles_issues),
    }

    return schema_ok and images_ok and smiles_ok, results
