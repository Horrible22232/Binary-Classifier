import torch

class Trainer:
    def __init__(self, config:dict, device:torch.device) -> None:
        """Initializes all needed training components.
        Args:
            config {dict} -- Configuration and hyperparameters of the trainer and model.
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        self.config = config
        self.device = device
    
    def run_training(self) -> None:
        pass
    
    def close(self) -> None:
        pass