import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, img_list, batch_size, img_size, img_channel=3):
        self.img_paths = np.array(img_list)
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_channel = img_channel
        self.data_count = 0  # Initialize data count
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.img_paths))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        batch_img = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.zeros((self.batch_size, self.img_channel, *self.img_size), dtype=np.uint8)

        for i, img_path in enumerate(batch_img):
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size)
            if img.shape[-1] != self.img_channel:
                img = img[..., :self.img_channel]

            # Print pixel values before normalization
            #print("Max Value (Before Normalization):", img.max())
            #print("Min Value (Before Normalization):", img.min())

            x[i] = np.moveaxis(img, -1, 0)  # Change the channel dimension to the front

        x = torch.tensor(x, dtype=torch.float32) / 255

        # Update data count for each batch
        self.data_count += len(x)

        # Print pixel values after normalization
        #print("Max Value (After Normalization):", x.max().item())
        #print("Min Value (After Normalization):", x.min().item())
        return x

def get_loader(train_input_dir, val_input_dir, batch_size=8, img_size=(3, 256, 256), seed=1234, img_channel=3, logger=None):
    train_img_paths = sorted(
        [os.path.join(train_input_dir, fname) for fname in os.listdir(train_input_dir) if fname.endswith(".jpg")]
    )

    val_img_paths = sorted(
        [os.path.join(val_input_dir, fname) for fname in os.listdir(val_input_dir) if fname.endswith(".jpg")]
    )

    train = DataGenerator(train_img_paths, batch_size, img_size, img_channel=img_channel)
    val = DataGenerator(val_img_paths, batch_size, img_size, img_channel=img_channel)

    if logger:
        logger.info("Data Loader is successfully loaded!")

    return train, val
