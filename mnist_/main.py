import torch.optim as optim
import torch.nn as nn
from model import Model
from torch.utils.tensorboard import SummaryWriter
from mnist_.config import Config
from mnist_.learningProcess import LearningProcess
# from faultInjector.hookSetter import HookSetter
from faultInjector.hookSetter import HookManager

import torch

torch.manual_seed(0)

if __name__ == "__main__":
    model = Model()

    hookManager = HookManager(model=model)
    hookManager.register_hooks(True)

    config = Config()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(config.pathToLogs)
    learner = LearningProcess(optimizer, criterion, writer)
    learner.train(model, hookManager)
    learner.validate(model)
    learner.test(model)