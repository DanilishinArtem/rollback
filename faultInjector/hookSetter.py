import torch
from faultInjector.hooks import GradientHook


class HookSetter:
    def __init__(self, model: torch.nn.Module):
        self.setter(model)
        self.hooks = {}
        self.backward_hooks = []

    def setter(self, model: torch.nn.Module):
        for name, layer in model.named_modules():
            # if hasattr(layer, "weight") or hasattr(layer, "bias"):
            Fhook = GradientHook(name, layer)
            self.hooks[name] = Fhook.hook
            self.backward_hooks.append(layer.register_backward_hook(Fhook.hook))
            print("registered hook for layer " + name)


class HookManager:
    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.gradient_hooks = {}

    def register_hooks(self, info):
        for name, layer in self.model.named_modules():
            if name not in self.gradient_hooks:
                self.gradient_hooks[name] = GradientHook(name, layer)
            hook_handle = layer.register_backward_hook(self.gradient_hooks[name].hook)
            self.hooks[name] = hook_handle
            if info:
                print("registered hook for layer " + name)

    def remove_hooks(self):
        for name, hook_handle in self.hooks.items():
            hook_handle.remove()
        self.hooks = {}