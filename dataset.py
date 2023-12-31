import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        self.mask_files = [file for file in os.listdir(mask_dir) if file.endswith('_segmentation.png')]

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_name)
        mask = Image.open(mask_name)

        # Ensure consistent sizing
        image = image.resize((256, 256), Image.ANTIALIAS)
        mask = mask.resize((256, 256), Image.ANTIALIAS)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def get_loader(image_dir, mask_dir, batch_size, num_workers, transform=None):
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = CustomDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return loader


if __name__ == "__main__":
    train_input_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1-2_Training_Input_x2"
    train_mask_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1_Training_GroundTruth_x2"
    batch_size = 4
    num_workers = 4

    # Define transformations here, if needed
    transform = transforms.Compose([transforms.ToTensor()])

    data_loader = get_loader(train_input_dir, train_mask_dir, batch_size, num_workers)

    for images, masks in data_loader:
        for i in range(batch_size):
            image = transforms.ToPILImage()(images[i])
            mask = transforms.ToPILImage()(masks[i])
            image.show()
            mask.show()
