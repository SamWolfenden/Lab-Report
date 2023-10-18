import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_loader, CustomDataset
from modules import UNet

class Train:
    def __init__(self, model, train_loader, learning_rate, num_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define loss and optimizer
        self.dice_loss = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_dice = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Calculate the iteration number within the epoch
                iteration = batch_idx

                # Forward pass
                outputs = self.model(inputs)

                # Calculate the Dice coefficient
                dice = self.calculate_dice(outputs, targets)
                total_dice += dice

                if torch.isnan(dice).any():
                    print("Dice coefficient contains NaN values.")
                    # Handle or log the issue and continue

                # Backpropagation
                self.optimizer.zero_grad()
                dice_loss = 1 - dice  # Invert Dice coefficient as it's a loss
                dice_loss.backward()
                self.optimizer.step()

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Iteration {iteration}, Dice Coefficient: {dice.item()}")

            average_dice = total_dice / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Dice Coefficient: {average_dice}")

    def calculate_dice(self, predicted, target, smooth=1e-5):
        predicted = torch.sigmoid(predicted)
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target, smooth=1e-5):
        predicted = torch.sigmoid(predicted)
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

if __name__ == '__main__':
    train_input_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1-2_Training_Input_x2"
    train_mask_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1_Training_GroundTruth_x2"
    batch_size = 4
    num_workers = 4
    learning_rate = 0.045
    num_epochs = 10

    # Create the data loader using your CustomDataset and get_loader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = get_loader(train_input_dir, train_mask_dir, batch_size, num_workers, transform=transform)

    model = UNet(1)

    trainer = Train(model, train_loader, learning_rate, num_epochs)
    trainer.train()
