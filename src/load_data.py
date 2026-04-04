import os
import csv
from PIL import Image

# class for loading data, which will be used in the training and evaluation of the model
class Dataset:
  # creates a dataset object by reading the image paths and the corresponding SMILES strings from a csv file, and applying any specified transformations to the images
  def __init__(self, image_dir, label_path, transform=None):
    self.image_dir = image_dir
    self.transform = transform
    self.samples = []
    
    with open(label_path, 'r') as f: 
      for line in csv.DictReader(f): # read the csv file and store the image path and the corresponding SMILES string in a list of dictionaries
        self.samples.append({"image": line["image"], "smiles": line["smiles"]})
  
  # returns the number of samples in the dataset      
  def __len__(self):
    return len(self.samples)
  
  # retrieves the image and the corresponding SMILES string for a given index, applies any specified transformations to the image, and returns both the image and the SMILES string as a tuple
  def __getitem__(self, idx): 
    sample = self.samples[idx]
    image_path = os.path.join(self.image_dir, sample["image"]) # constructs full path 
    image = Image.open(image_path).convert("RGB") # opens image from path as RGB
    
    if self.transform: 
      image = self.transform(image) 
      
    return image, sample["smiles"] 
  