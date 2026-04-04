import os
import csv
from PIL import Image


# class for loading data, which will be used in the training and evaluation of the model
class Dataset:
    # creates a dataset object by reading the image paths and the corresponding SMILES strings from a csv file, and applying any specified transformations to the images
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.label_path = label_path
        self.transform = transform
        self.samples = []

        with open(label_path, 'r', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # basic check to ensure CSV has required columns
            if reader.fieldnames is None:
                raise ValueError("CSV file is missing a header row.")

            required_columns = {"image", "smiles"}
            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            # read the csv file and store the image path and the corresponding SMILES string in a list of dictionaries
            for line in reader:
                image_name = line["image"].strip()
                smiles = line["smiles"].strip()

                self.samples.append({
                    "image": image_name,
                    "smiles": smiles
                })

    # returns the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # retrieves the image and the corresponding SMILES string for a given index, applies any specified transformations to the image, and returns both the image and the SMILES string as a tuple
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # constructs full path
        image_path = os.path.join(self.image_dir, sample["image"])

        # ensures the image file exists before attempting to open
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # opens image from path as RGB
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, sample["smiles"]
