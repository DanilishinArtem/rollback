import torch.optim as optim
import torch.nn as nn
from model import Model
from config import Config
from learningProcess import LearningProcess
from config import Config
import time
import torch
from hook import HookManager
from torch import nn

torch.manual_seed(20)


if __name__ == "__main__":
    model = Model()
    HookManager(model).register_hooks(True)

    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    learner = LearningProcess(optimizer, criterion)
    learner.train(model)
    # learner.validate(model)
    # learner.test(model)
