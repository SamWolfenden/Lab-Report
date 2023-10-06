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

    print(f"Number of samples in the dataset: {len(dataset)}")

    sample_idx = 1999  # Change the index to display different samples

    if sample_idx < len(dataset):
        sample_image, sample_label = dataset[sample_idx]

        # Convert the tensor to a NumPy array and transpose the dimensions
        sample_image = sample_image.permute(1, 2, 0).numpy()

        # Plot the image and superpixel label
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_image)
        plt.title('Sample Image')
        plt.axis('off')

        # Print the filename underneath the image
        plt.text(0, -20, dataset.image_files[sample_idx], fontsize=10, color='black')

        if sample_label is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(sample_label.squeeze() if sample_label is not None else None, cmap='viridis')
            plt.title('Superpixel Label')
            plt.axis('off')

            # Print the filename underneath the label
            plt.text(0, -20, dataset.label_files[sample_idx], fontsize=10, color='black')

        plt.show()
    else:
        print(f"Invalid sample index. The dataset contains {len(dataset)} samples.")
"""
