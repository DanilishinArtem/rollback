import torch
import torch.optim as optim
import numpy as np
from detector.config import ConfigDetecotr
from faultInjector.hookSetter import HookManager

config = ConfigDetecotr()


class ExpMovingAverages:
    def __init__(self, optimizer, beta1=0.9, beta2=0.999):
        startBound = 1
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
                    self.exp_avg_sqs.append(torch.ones_like(param.data) * startBound)
    
    def update(self):
        idx = 0
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    grad = param.grad.data
                    self.exp_avgs[idx].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                    self.exp_avg_sqs[idx].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                    idx += 1

    def detectAndCorrect(self, step, correct: bool):
        # self.step += 1
        anomalies = {}
        idx = 0
        faults = 0
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    upperBound = self.exp_avgs[idx] / bias_correction1 + self.k * torch.sqrt(self.exp_avg_sqs[idx] / bias_correction2)
                    lowerBound = self.exp_avgs[idx] / bias_correction1 - self.k * torch.sqrt(self.exp_avg_sqs[idx] / bias_correction2)
                    out_of_bounds = torch.logical_or(param.grad.data < lowerBound, param.grad.data > upperBound)
                    if out_of_bounds.any():
                        faults += torch.sum(out_of_bounds).item()
                        if correct:
                          anomalies[idx] = out_of_bounds.nonzero(as_tuple=True)
                          param.grad.data[out_of_bounds] = self.generate_normal_tensor_within_bounds(lowerBound[out_of_bounds], upperBound[out_of_bounds], param.grad.data[out_of_bounds])
                    idx += 1
        if faults > 0:
            print("[Detected]: total number of faults = " + str(faults))
        return faults

    def generate_normal_tensor_within_bounds(self, lower_bound, upper_bound, size):
        mean = (upper_bound + lower_bound) / 2
        std = (upper_bound - lower_bound) / 6
        tensor = torch.normal(mean, std)
        while True:
            mask = (tensor >= lower_bound) & (tensor <= upper_bound)
            if torch.all(mask):
                break
            tensor = torch.where(mask, tensor, torch.normal(mean, std))
        return tensor

    def numberFaults_(self, hookManager: HookManager, model, optimizer, criterion, input, output):
        gradsWithoutFaults = []
        gradsWithFaults = []
        hookManager.freeze()
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
        # hookManager.register_hooks(False)
        hookManager.set_hooks()
        # find number of deffirences between gradsWithFaults and gradsWithoutFaults
        faults = 0
        for i in range(len(gradsWithoutFaults)):
            faults += (gradsWithFaults[i] != gradsWithoutFaults[i]).sum().item()
        return faults

    def numberFaults(self, hookManager: HookManager, model, optimizer, criterion, input, output):
        gradsWithoutFaults = []
        gradsWithFaults = []
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
        # hookManager.register_hooks(False)
        hookManager.set_hooks()
        # find number of deffirences between gradsWithFaults and gradsWithoutFaults
        faults = 0
        for i in range(len(gradsWithoutFaults)):
            faults += (gradsWithFaults[i] != gradsWithoutFaults[i]).sum().item()
        return faults

    def correct(self):
        pass    

    
