import torch
from torch import optim
from torch import nn
from model import Classifier
from datagen import SinusGenerator 

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
        self.data_gen = SinusGenerator(self.batch_size)  
    
    def run_training(self) -> None:
        """
        Trains the model for the specified number of epochs.
        """
        for (epoch, (batch, label)) in enumerate(self.data_gen.sample(self.epochs)):
            # Convert the data to a tensor and move it to the device
            batch, label = torch.tensor(batch, dtype=torch.float32).to(self.device), torch.tensor(label, dtype=torch.float32).to(self.device)
            # Calculate the loss and optimize the model
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            
            # Print the loss and evaluation score
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            self.evaluate()
    
    def evaluate(self) -> None:
        """
        Evaluates the model on the test set.
        """
        # Get the test set
        test_batch, label = list(self.data_gen.sample(1))[0]
        # Convert the data to a tensor
        test_batch, label = torch.tensor(test_batch, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # Get the output of the model
        output = self.model(test_batch).to(self.device)
        # Get the prediction prob
        output = torch.sigmoid(output).detach().cpu()
        # Calculate the accuracy
        accuracy_label_0, accuracy_label_1 = 1. - output[label == 0].mean().item(), output[label == 1].mean().item()
        print("Acurracy for label 0: {:2f}, Accuracy score for label 1: {:2f}".format(accuracy_label_0, accuracy_label_1))
    
    def close(self) -> None:
        pass