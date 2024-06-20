import torch
import torch.optim as optim
import numpy as np
from detector.config import ConfigDetecotr
from faultInjector.hookSetter import HookManager

config = ConfigDetecotr()


class ExpMovingAverages:
    def __init__(self, optimizer, beta1=0.9, beta2=0.999):
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.exp_avgs = []
        self.exp_avg_sqs = []
        self.k = np.sqrt(1/config.p_fault)
        self.step = 0
        
        # Инициализация exp_avgs и exp_avg_sqs для каждого параметра
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.exp_avgs.append(torch.zeros_like(param.data))
                    self.exp_avg_sqs.append(torch.ones_like(param.data) * 0.01)
    
    def update(self):
        idx = 0
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    grad = param.grad.data
                    self.exp_avgs[idx].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                    self.exp_avg_sqs[idx].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                    idx += 1

    def detect(self):
        self.step += 1
        anomalies = {}
        idx = 0
        faults = 0
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    upperBound = self.exp_avgs[idx] / bias_correction1 + self.k * torch.sqrt(self.exp_avg_sqs[idx] / bias_correction2)
                    lowerBound = self.exp_avgs[idx] / bias_correction1 - self.k * torch.sqrt(self.exp_avg_sqs[idx] / bias_correction2)
                    out_of_bounds = torch.logical_or(param.grad.data < lowerBound, param.grad.data > upperBound)
                    if out_of_bounds.any():
                        faults += torch.sum(out_of_bounds).item()
                        # anomalies[idx] = out_of_bounds.nonzero(as_tuple=True)
                    idx += 1
        if faults > 0:
            print("[Detected]: total number of faults = " + str(faults))
        return faults

    def numberFaults(self, hookManager: HookManager, model, optimizer, criterion, input, output):
        gradsWithoutFaults = []
        gradsWithFaults = []
        # calculate gradients with faults
        optimizer.zero_grad()
        result = model(input)
        loss = criterion(result, output)
        loss.backward()
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    gradsWithFaults.append(param.grad.data)
        # turn off hooks
        hookManager.remove_hooks()
        # calculate gradients without faults
        optimizer.zero_grad()
        result = model(input)
        loss = criterion(result, output)
        loss.backward()
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    gradsWithoutFaults.append(param.grad.data)
        # register hooks again
        hookManager.register_hooks(False)
        # find number of deffirences between gradsWithFaults and gradsWithoutFaults
        faults = 0
        for i in range(len(gradsWithoutFaults)):
            if not torch.equal(gradsWithoutFaults[i], gradsWithFaults[i]):
                    faults += 1
        return faults


    def correct(self):
        pass    

    
