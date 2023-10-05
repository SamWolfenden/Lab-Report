import os
from torch.utils.data import Dataset
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform

        # Define the root directory based on the provided split
        if split == 'train':
            self.root_dir = r'C:\Users\sam\Downloads\ISIC-2017_Training_Data'
        elif split == 'val':
            self.root_dir = r'C:\Users\sam\Downloads\ISIC-2017_Validation_Data'
        elif split == 'test':
            self.root_dir = r'C:\Users\sam\Downloads\ISIC-2017_Test_v2_Data'
        else:
            raise ValueError()

        # Determine the directory based on the provided split and root_dir
        self.data_dir = os.path.join(self.root_dir, split)

        # List all image files in the data directory
        self.image_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
