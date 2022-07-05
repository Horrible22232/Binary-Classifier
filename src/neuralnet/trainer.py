import torch
from torch import optim
from torch import nn
from model import Classifier
from datagen import SinusGenerator 

class Trainer:
    def __init__(self, config:dict, device:torch.device) -> None:
        """Initializes all needed training components.
        Arguments:
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
        self.train_data_gen = SinusGenerator(self.batch_size)
        self.test_data_gen = SinusGenerator(500) 
    
    def run_training(self) -> None:
        """
        Trains the model for the specified number of epochs.
        """
        for epoch, (batch, label) in enumerate(self.train_data_gen.sample(self.epochs)):
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
        test_batch, label = list(self.test_data_gen.sample(1))[0]
        # Convert the data to a tensor
        test_batch, label = torch.tensor(test_batch, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # Get the output of the model
        output = self.model(test_batch).to(self.device)
        # Get the prediction probability
        output = torch.sigmoid(output).detach().cpu()
        # Get the predicted label
        pred_label = torch.bernoulli(output)
        # Calculate the true positive and true negative rate
        true_positive, true_negative = pred_label[label == 1].sum() / label.sum(), (1. - pred_label[label == 0]).sum() / (1. - label).sum()
        # Calculate the accuracy
        accuracy = (pred_label == label).sum() / label.shape[0]
        # Print the results
        print("True positive score: {:2f}, True negative score: {:2f}, Accuracy: {:2f}".format(true_positive, true_negative, accuracy))
    
    def close(self) -> None:
        pass