"""
Data validation functions for the chemical image recognition project.
"""

import os
import csv
from rdkit import Chem


# Validate the CSV schema of the labels file
def validate_csv_schema(label_path):
  """
  Validate the CSV schema of the labels file.
  Checks for required columns.
  """
  issues = []

  with open(label_path, 'r', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # check header exists
    if reader.fieldnames is None:
      issues.append("CSV file is missing a header row.")
      return False, issues

    # check required columns
    required_columns = {"image", "smiles"}
    missing = required_columns - set(reader.fieldnames)

    if missing:
      issues.append(f"CSV missing required columns: {missing}")

  return len(issues) == 0, issues


# checks to see if image listed in csv exists
def validate_images(image_dir, label_path):
  """
  Validate that all images listed in the labels file exist in the image directory.
  """
  issues = []

  with open(label_path, 'r', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, start=1):
      image_name = row["image"].strip()
      image_path = os.path.join(image_dir, image_name)

      if not os.path.exists(image_path):
        issues.append(f"Row {i}: missing image file -> {image_path}")

  return len(issues) == 0, issues


# checks whether each SMILES string is valid
def validate_smiles(label_path):
  """
  Validate the SMILES strings in the labels file.
  Checks for valid SMILES strings using RDKit.
  """
  issues = []

  with open(label_path, 'r', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, start=1):
      smiles = row["smiles"].strip()
      mol = Chem.MolFromSmiles(smiles)

      if mol is None:
        issues.append(f"Row {i}: invalid SMILES string -> {smiles}")

  return len(issues) == 0, issues


# runs all validation checks
def validate_data(image_dir, label_path):
  """
  Run all data validation checks.
  """
  results = {}

  schema_ok, schema_issues = validate_csv_schema(label_path)
  images_ok, image_issues = validate_images(image_dir, label_path)
  smiles_ok, smiles_issues = validate_smiles(label_path)

  results["schema"] = (schema_ok, schema_issues)
  results["images"] = (images_ok, image_issues)
  results["smiles"] = (smiles_ok, smiles_issues)

  overall_ok = schema_ok and images_ok and smiles_ok

  return overall_ok, results
