"""
Validate SMILES strings in the dataset.

goes through a dataset and checks tos ee if SMILES can be parsed by RDKIT (for debugging) 
"""


from rdkit import Chem
from src.load_data import Dataset
from test.utils import resolve_dataset_paths

image_dir, label_path = resolve_dataset_paths()

dataset = Dataset(
    image_dir=image_dir,
    label_path=label_path
)

valid_count = 0
invalid_count = 0
invalid_examples = []

# iterate through the dataset and validate each SMILES string using RDKit
for i in range(len(dataset)):
    _, smiles = dataset[i]
    # returns None if the SMILES string is invalid, otherwise returns a molecule object
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        valid_count += 1  # if invalid smiles string, but it shouldn't occur
    else:
        invalid_count += 1
        invalid_examples.append(smiles)

print("Total samples:", len(dataset))
print("Valid SMILES:", valid_count)
# should not happen but we include it incase
print("Invalid SMILES:", invalid_count)

if invalid_examples:
    print("Invalid examples:")
    for s in invalid_examples:
        print(" ", s)
