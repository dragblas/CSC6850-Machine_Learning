from torchvision import transforms

# gets transformed data
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
