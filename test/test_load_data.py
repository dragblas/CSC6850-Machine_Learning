"""
    script loads dataset, and applies transofmration, 
    prints sample to confirm images and smiles labels are loaded
    convert into tnesors
"""

from typing import cast
from torch import Tensor
from src.load_data import Dataset
from src.transforms import get_default_transform
from test.utils import resolve_dataset_paths

image_dir, label_path = resolve_dataset_paths()

# creates an instance of Dataset class using test data
dataset = Dataset(
    image_dir=image_dir,
    label_path=label_path,
    transform=get_default_transform()
)

# prints size of dataset
print("Dataset size:", len(dataset))

# prints first few samples
for i in range(min(5, len(dataset))):
    image, smiles = dataset[i]
    image = cast(Tensor, image) # cast to Tensor for type checking

    print(f"Sample {i}")
    print("SMILES:", smiles)
    print("Tensor shape:", image.shape)
