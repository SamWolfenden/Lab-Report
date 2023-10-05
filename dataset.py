import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class ISICDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Determine the directory based on the provided root_dir
        self.data_dir = root_dir

        # List all image files in the data directory
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]

        # List all superpixel label files in the data directory
        self.label_files = [f for f in os.listdir(self.data_dir) if f.endswith('_superpixels.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        label_name = os.path.join(self.data_dir, self.label_files[idx])

        image = Image.open(img_name)
        label = Image.open(label_name).convert('L')  # Convert label to grayscale

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

"""
if __name__ == "__main__":
    # Define the directory where the ISIC dataset is located
    dataset_root = r'C:\Users\sam\Downloads\ISIC-2017_Training_Data'

    # Create a transform for image preprocessing
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])

    # Create an instance of the ISICDataset class
    dataset = ISICDataset(root_dir=dataset_root, transform=transform)

    # Display a sample image and its corresponding superpixel label
    sample_idx = 0  # Change the index to display different samples
    sample_image, sample_label = dataset[sample_idx]

    # Convert the tensor to a NumPy array for display
    sample_image = sample_image.permute(1, 2, 0).numpy()

    # Plot the image and superpixel label
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image)
    plt.title('Sample Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sample_label.squeeze(), cmap='viridis')  # Squeeze the label tensor to 2D
    plt.title('Superpixel Label')
    plt.axis('off')

    plt.show()
"""