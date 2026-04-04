"""
Validate SMILES strings in the dataset.
"""


from rdkit import Chem
from src.load_data import Dataset

dataset = Dataset(
  image_dir="data/test/images",
  label_path="data/test/labels.csv"
)

valid_count = 0
invalid_count = 0
invalid_examples = []

# iterate through the dataset and validate each SMILES string using RDKit
for i in range(len(dataset)):
  _, smiles = dataset[i] 
  mol = Chem.MolFromSmiles(smiles) # returns None if the SMILES string is invalid, otherwise returns a molecule object

  if mol is not None:
    valid_count += 1
  else:
    invalid_count += 1
    invalid_examples.append(smiles)

print("Total samples:", len(dataset))
print("Valid SMILES:", valid_count)
print("Invalid SMILES:", invalid_count)

if invalid_examples:
  print("Invalid examples:")
  for s in invalid_examples:
    print(" ", s)