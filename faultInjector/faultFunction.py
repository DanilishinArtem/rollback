import torch
from faultInjector.config import ConfigInjector

config = ConfigInjector()



class FaultFunction:
    @staticmethod
    def impulsFunction(input: torch.tensor):
        faultValue = config.faultValue
        random_indices = torch.randint(0, input.numel(), (config.numFaults,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        input[random_positions] = faultValue

    @staticmethod
    def randomFunction(input: torch.tensor):
        faultsN = config.numFaults
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        input[random_positions] = torch.rand_like(input[random_positions])

    @staticmethod
    def zeroFunction(input: torch.tensor):
        faultsN = config.numFaults
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        input[random_positions] = 0

    @staticmethod
    def valueFunction(input: torch.tensor):
        faultValue = config.faultValue
        faultsN = config.numFaults
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        input[random_positions] = faultValue

    @staticmethod
    def magnitudeFunction(input: torch.tensor):
        faultValue = config.faultValue
        faultsN = config.numFaults
        random_indices = torch.randint(0, input.numel(), (faultsN,))
        random_positions = torch.unravel_index(random_indices, input.shape)
        input[random_positions] = input[random_positions] * faultValue