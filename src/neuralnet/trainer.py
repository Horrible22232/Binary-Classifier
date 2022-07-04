import torch
from torch import optim
from torch import nn
from model import Classifier
from datagen import SinEnv 

class Trainer:
    def __init__(self, config:dict, device:torch.device) -> None:
        """Initializes all needed training components.
        Args:
            config {dict} -- Configuration and hyperparameters of the trainer and model.
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        self.config = config
        self.device = device
        self.lr = self.config['learning_rate']
        self.epochs = self.config['epochs']
        self.batch_size = self.config['batch_size']
        self.model = Classifier(2, self.config, self.device).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.test_env = SinEnv(self.batch_size)  
    
    def run_training(self) -> None:
        for (epoch, (batch, label)) in enumerate(self.test_env.sample(self.epochs)):
            self.optimizer.zero_grad()
            output = self.model(torch.tensor(batch, dtype=torch.float32).to(self.device))
            loss = self.criterion(output, torch.tensor(label, dtype=torch.float32).to(self.device))
            loss.backward()
            self.optimizer.step()
            
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            self.evaluate()
    
    def evaluate(self) -> None:
        for (test_batch, label) in self.test_env.sample(1):
            output = self.model(torch.tensor(test_batch, dtype=torch.float32).to(self.device))
            output = torch.sigmoid(output).detach().cpu()
            label = torch.tensor(label, dtype=torch.float32)
            
            print("Probality for label 0: {:2f}, Probability for label 1: {:2f}".format(output[label == 0].mean().item(), output[label == 1].mean().item()))
    
    def close(self) -> None:
        pass