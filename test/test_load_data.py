from torchvision import transforms
from src.load_data import Dataset

# transformation pipeline for resizing images and converting them to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# creates an instance of Dataset class using test data from data/test/
dataset = Dataset(
    image_dir="data/test/images",
    label_path="data/test/labels.csv",
    transform=transform
)

# prints size of dataset
print("Dataset size:", len(dataset))

# prints first image and corresponding SMILES string in the dataset
for i in range(min(5, len(dataset))):
    image, smiles = dataset[i]
    print(f"Sample {i}")
    print("SMILES:", smiles)
    print("Tensor shape:", image.shape)
