from src.load_data import Dataset

# creates an instance of Dataset class using test data from data/test/ 
dataset = Dataset(image_dir="data/test/images", label_path="data/test/labels.csv")

# prints size of dataset 
print("Dataset size:", len(dataset))  

# prints first image and corresponding SMILES string in the dataset
for i in range(min(5, len(dataset))):
  image, smiles = dataset[i]
  print(f"Sample {i}")
  print("SMILES:", smiles)
  print("Image size:", image.size)