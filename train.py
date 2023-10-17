import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_loader, DataGenerator
from modules import UNet

class Train:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.045, num_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, data in enumerate(self.train_loader):
                inputs = data.to(self.device)
                labels = inputs

                # Calculate the iteration number within the epoch
                iteration = batch_idx

                # Forward pass
                outputs = self.model(inputs)

                # Compute the Cross-Entropy loss
                loss = self.criterion(outputs, labels)

                if torch.isnan(loss).any():
                    print("Loss contains NaN values.")
                    # Handle or log the issue and continue

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Iteration {iteration}, Loss: {loss.item()}")

                # Break the loop after the correct number of iterations
                if iteration == math.floor((len(train_loader)/self.train_loader.batch_size)):
                    break

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {total_loss / (len(train_loader)/self.train_loader.batch_size)}")


if __name__ == '__main__':
    train_input_dir = r'C:\Users\sam\Downloads\ISIC-2017_Training_Data'
    val_input_dir = r'C:\Users\sam\Downloads\ISIC-2017_Validation_Data'
    #train_input_dir = r'/home/Student/s4748611/Lab-Report/ISIC-2017_Training_Data'
    #val_input_dir = r'/home/Student/s4748611/Lab-Report/ISIC-2017_Validation_Data'

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_loader, val_loader = get_loader(train_input_dir, val_input_dir, batch_size=32, img_size=(64, 64), seed=1234)

    model = UNet(3,3)
    learning_rate = 0.001
    num_epochs = 10

    trainer = Train(model, train_loader, val_loader, learning_rate, num_epochs)
    trainer.train()
