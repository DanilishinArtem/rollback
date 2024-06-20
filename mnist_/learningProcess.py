from torch.utils.data import DataLoader, random_split
from dataLoader import load_mnist_dataset
from config import Config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from detector.infoGetteer import ExpMovingAverages
from faultInjector.hookSetter import HookManager


class LearningProcess:
    def __init__(self, optimizer: optim, criterion: nn.Module, writer: SummaryWriter = None):
        self.config = Config()
        self.writer = writer
        self.train_loader, self.val_loader, self.test_loader = self.createDataset()
        self.optimizer = optimizer
        self.criterion = criterion
        self.detector = ExpMovingAverages(self.optimizer)

    def createDataset(self):
        dataset = load_mnist_dataset()
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_size = int(self.config.split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train(self, model: nn.Module, hookManager: HookManager):
        total_counter = 1
        total_loss = 0
        for epoch in range(self.config.num_epochs):
            model.train()
            for batch in self.train_loader:
                images, labels = batch["image"], batch["label"]

                numberFaults = self.detector.numberFaults(hookManager, model, self.optimizer, self.criterion, images, labels)

                self.optimizer.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                
                detected_numberFaults = self.detector.detect()

                if detected_numberFaults == 0:
                    # implement rallback method for case > 0
                    self.detector.update()

                self.optimizer.step()
                total_loss += loss.item()
                total_counter += 1

                print("For step " + str(total_counter) + " training loss = " + str(round(total_loss / total_counter,2)) + ", number of faults = " + str(numberFaults) + ", detected number of faults = " + str(detected_numberFaults))

    def validate(self, model: nn.Module):
        total_counter = 0
        model.eval()
        val_loss = 0
        correct = 0
        batch_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                val_loss += self.criterion(output, labels).item()
                batch_loss = self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                self.writer.add_scalar("Loss/validation", batch_loss / self.config.batch_size, total_counter)
        val_loss /= len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)
        print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")

    def test(self, model: nn.Module):
        model.eval()
        test_loss = 0
        correct = 0
        counter = 0
        total_counter = 0
        batch_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                total_counter += 1
                images, labels = batch["image"], batch["label"]
                output = model(images)
                test_loss += self.criterion(output, labels).item()
                batch_loss = self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                self.writer.add_scalar("Loss/test", batch_loss / self.config.batch_size, total_counter)

        test_loss /= len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)
        print(f"Test Loss: {test_loss}, Test Accuracy: {accuracy}")
