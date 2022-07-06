import os
import pickle
import torch
from torch import optim
from torch import nn
from model import Classifier
from utils import create_data_loader

class Trainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.
        Arguments:
            config {dict} -- Configuration and hyperparameters of the trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        self.config = config
        self.device = device
        self.run_id = run_id
        self.lr = self.config['learning_rate']
        self.epochs = self.config['epochs']
        self.batch_size = self.config['batch_size']
        self.model = Classifier(2, self.config, self.device).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data_gen = create_data_loader(config)
        self.test_data_gen = create_data_loader(config) 
    
    def run_training(self) -> None:
        """
        Trains the model for the specified number of epochs.
        """
        for epoch, (batch, label) in enumerate(self.train_data_gen.sample(self.epochs, self.batch_size)):
            # Convert the data to a tensor and move it to the device
            batch, label = torch.tensor(batch, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.float32, device=self.device)
            # Calculate the loss and optimize the model
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            
            # Print the loss and evaluation score
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            self.evaluate()
        # Save the model and the used training config after the training
        self._save_model()
    
    def evaluate(self) -> None:
        """
        Evaluates the model on the test set.
        """
        # Get the test set
        test_batch, label = list(self.test_data_gen.sample(num_batches=1, num_samples=1000))[0]
        # Convert the data to a tensor
        test_batch, label = torch.tensor(test_batch, dtype=torch.float32, device=self.device), torch.tensor(label, dtype=torch.float32, device=self.device)
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
    
    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")
    
    def close(self) -> None:
        pass