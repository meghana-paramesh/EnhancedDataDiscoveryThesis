import torchvision.transforms as transforms

def get_augmentations():
# Example of minor augmentations
    return transforms.Compose([
        transforms.RandomCrop(size=(224, 224)),       # Random crop to 224x224 size
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Color jittering
        transforms.GaussianBlur(kernel_size=3),       # Random Gaussian blur with kernel size 3
        transforms.ToTensor(),                        # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Image normalization
    ])

