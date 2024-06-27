import torch
from torch import nn


class GradientHook:
    def __init__(self, name: str, model: torch.nn.Module):
        self.name = name
        self.model = model

    def full_backward_pre_hook(self, module, grad_output):
        print("info about model on the backward method")
        print("name of layer: " + self.name)
        print(grad_output)
        print("\n")
        for param in self.model.named_parameters():
            if param[1].grad is not None:
                print("info about model on the backward method")
                print("name of layer: " + param[0])
                print(param[1].grad.shape)
                print("\n")


    def full_backward_hook(self, module, grad_input, grad_output):
        print("info about model on the backward method")
        print("name of layer: " + self.name)
        # print(grad_output[0].shape)
        print(module.weight.grad)
        print("\n")
        for param in self.model.named_parameters():
            if param[0] != "conv1" and param[1].grad is not None:
                print("info about model on the backward method")
                print("name of layer: " + param[0])
                print(param[1].grad.shape)
                print("\n")



class HookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.gradient_hooks = {}

    def register_hooks(self, info):
        for layer in self.model.modules():
            name = layer._get_name()
            if name != "Model":
                print("name: " + name)
                self.gradient_hooks[name] = GradientHook(name, self.model)
                # register_full_backward_hook
                # register_full_backward_pre_hook
                # register_backward_hook
                hook_handle = layer.register_full_backward_hook(self.gradient_hooks[name].full_backward_hook)
                self.hooks[name] = hook_handle
                if info:
                    print("registered hook for layer " + name)
                break