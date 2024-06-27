from torch.utils.data import DataLoader, random_split
from dataLoader import load_mnist_dataset
from config import Config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import time


class LearningProcess:
    def __init__(self, optimizer: optim, criterion: nn.Module):
        self.train_loader, self.val_loader, self.test_loader = self.createDataset()
        self.optimizer = optimizer
        self.criterion = criterion

    def createDataset(self):
        dataset = load_mnist_dataset()
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_size = int(Config.split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train(self, model: nn.Module):
        print("start training\n\n")
        start = time.time()
        total_counter = 0
        total_loss = 0
        correct = 0
        numPic = 0
        # hookManager.remove_hooks()
        for epoch in range(Config.num_epochs):
            model.train()
            for batch in self.train_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                self.optimizer.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                numPic += len(images)
                print("\n")
                print("=============Start of backward===============")
                loss.backward()
                print("==============End of backward================")
                print("\n")
                self.optimizer.step()
                total_loss += loss.item()
                print("For step " + str(total_counter) + " training loss = " + str(round(total_loss / total_counter,2)) + " training accuracy = " + str(round(correct / numPic,2)))
                
        print("Training time: " + str(round(time.time() - start,2)) + " sec.")

    def validate(self, model: nn.Module):
        total_counter = 0
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in self.val_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                val_loss += self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")

    def test(self, model: nn.Module):
        model.eval()
        test_loss = 0
        correct = 0
        total_counter = 0
        with torch.no_grad():
            for batch in self.test_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                test_loss += self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {accuracy}")
