"""
 generates molecule images from csv 

 converts csv containing SMILES strings into a dataset with images and a labels.csv 
"""

import os
import csv
from rdkit import Chem
from rdkit.Chem import Draw


def generate_dataset(input_csv, smiles_column, output_dir):
  images_dir = os.path.join(output_dir, "images")
  labels_path = os.path.join(output_dir, "labels.csv")

  os.makedirs(images_dir, exist_ok=True)

  count_total = 0
  count_valid = 0
  count_invalid = 0

  with open(input_csv, "r", encoding="utf-8") as infile, \
      open(labels_path, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    writer.writerow(["image", "smiles"])

    for row in reader:
      count_total += 1

      smiles = row[smiles_column].strip()
      mol = Chem.MolFromSmiles(smiles)

      if mol is None:
        count_invalid += 1
        continue

      image_name = f"img_{count_valid:04d}.png"
      image_path = os.path.join(images_dir, image_name)

      img = Draw.MolToImage(mol)
      img.save(image_path)

      writer.writerow([image_name, smiles])
      count_valid += 1

  print("Dataset generation complete.")
  print("Total:", count_total)
  print("Valid:", count_valid)
  print("Invalid:", count_invalid)
