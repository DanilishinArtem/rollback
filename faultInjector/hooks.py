import torch
from faultInjector.config import ConfigInjector
from faultInjector.faultFunction import FaultFunction

config = ConfigInjector()

def functionSetter(input: torch.tensor):
    input = input[0]
    modified_input = input.clone()
    faultFunction = config.function
    if faultFunction == "impulsFunction":
        FaultFunction.impulsFunction(modified_input)
    if faultFunction == "randomFunction":
        FaultFunction.randomFunction(modified_input)
    if faultFunction == "zeroFunction":
        FaultFunction.zeroFunction(modified_input)
    if faultFunction == "valueFunction":
        FaultFunction.valueFunction(modified_input)
    if faultFunction == "magnitudeFunction":
        FaultFunction.magnitudeFunction(modified_input)
    return modified_input

class GradientHook:
    def __init__(self, name: str, layer: torch.nn.Module):
        self.name = name
        self.layer = layer
        self.counter = 0
        self.duration = 0

    def hook(self, module, grad_input, grad_output):
        self.counter += 1
        if self.counter >= config.startFault and self.duration <= config.duration and self.name in config.nameLayer:
            self.duration += 1
            print("fault (gradient) for layer " + self.name + " was injected at time " + str(self.counter))
            modified_grad_input = functionSetter((grad_input[0],))
            return (modified_grad_input,) + grad_input[1:]