"""
This script generates a small test dataset of molecule images and their corresponding SMILES strings.
It uses the RDKit library to create images from SMILES strings and saves them in a specified directory.
"""


from rdkit import Chem
from rdkit.Chem import Draw
import os
import csv

output_dir = "data/test/images" 
os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist

# sample test data 
data = [
  ("ethanol.png", "CCO"),           # ethanol
  ("benzene.png", "c1ccccc1"),     # benzene
  ("acetic_acid.png", "CC(=O)O"),      # acetic acid
  ("ethylamine.png", "CCN"),          # ethylamine
  ("ether.png", "CCOCC"),        # ether
  ("bad1.png", "C1CC"),
  ("bad2.png", "C(=O"),
  ("bad3.png", "CC))O"),
  ("bad4.png", "ZC"),
  ("bad5.png", "C=C==C"),
]

# path to save the CSV file containing image filenames and their corresponding SMILES strings
# using test directory for now
csv_path = "data/test/labels.csv" 

# generate images and save the CSV file
with open(csv_path, "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(["image", "smiles"])

  for filename, smiles in data:
    mol = Chem.MolFromSmiles(smiles) # create molecule object from SMILES string
    img = Draw.MolToImage(mol) # generate image from molecule object

    img.save(os.path.join(output_dir, filename)) # save image to output directory
    writer.writerow([filename, smiles]) # write the image filename and corresponding SMILES string to the CSV file

print("Generated test dataset.")