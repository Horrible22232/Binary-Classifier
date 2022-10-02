import os
import time
import pickle
import torch
from torch import nn
from torch import optim
from model import Classifier
from utils import create_data_loader
from torch.utils.tensorboard import SummaryWriter

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
        self.updates = self.config['updates']
        self.batch_size = self.config['batch_size']
        self.train_data_gen = create_data_loader(config["data"])
        # self.test_data_gen = create_data_loader(config["data"])
        self.model = Classifier(self.train_data_gen.dim, self.config["model"]).to(self.device)
        self.model.train()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup Tensorboard Summary Writer
        if not os.path.exists("./summaries"):
            os.makedirs("./summaries")
        timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
        self.writer = SummaryWriter("./summaries/" + run_id + timestamp)
    
    def run_training(self) -> None:
        """Trains the model for the specified number of epochs.
        """
        for update, data in enumerate(self.train_data_gen.sample(self.updates, self.batch_size)):
            # Get the samples and labels as a tensor and move it to the device
            samples, label = torch.tensor(data["samples"], dtype=torch.float32, device=self.device), torch.tensor(data["labels"], dtype=torch.float32, device=self.device)
            # Calculate the loss and optimize the model
            self.optimizer.zero_grad()
            output = self.model(samples)
            # Mask the output if necessary
            if "masks" in data:
                output = output[data["masks"] == 1]
            loss = self.criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Evaluate the model on the test set
            true_positive, true_negative, accuracy = self.evaluate(data)
            
            # Print the loss and evaluation score
            print("Update: {}, Loss: {}".format(update, loss.item()), flush=True)
            print("True positive score: {:2f}, True negative score: {:2f}, Accuracy: {:2f}".format(true_positive, true_negative, accuracy), flush=True)
            
            # Write the training statistics to the summary file
            training_stats = {"loss": loss.item(), "true_positive": true_positive, "true_negative": true_negative, "accuracy": accuracy}
            self._write_training_summary(update, training_stats)
            
            if update % self.config["save_iter"] == 0:
                self._save_model()
            
        # Save the model and the used training config after the training
        self._save_model()
    
    def evaluate(self, data) -> tuple:
        """Evaluates the model on the train set.
        Returns:
            {tuple} -- A tuple containing the true positive, true negative and accuracy scores.
        """
        # Get the test data
        # data = next(self.test_data_gen.sample(num_batches=1, num_samples=1000))
        # Get the samples and labels as a tensor
        test_samples, label = torch.tensor(data["samples"], dtype=torch.float32, device=self.device), torch.tensor(data["labels"], dtype=torch.float32)
        # Get the output of the model
        output = self.model(test_samples)
        # Mask the output if necessary
        if "masks" in data:
            output = output[data["masks"] == 1]
        # Get the prediction probability
        output = torch.sigmoid(output).detach().cpu()
        # Get the predicted label
        pred_label = torch.bernoulli(output)
        # Calculate the true positive and true negative rate
        true_positive, true_negative = pred_label[label == 1].sum() / label.sum(), (1. - pred_label[label == 0]).sum() / (1. - label).sum()
        # Calculate the accuracy
        accuracy = (pred_label == label).sum() / label.shape[0]     
        
        return true_positive, true_negative, accuracy
    
    def _write_training_summary(self, update, training_stats) -> None:
        """Writes to an event file based on the run-id argument.
        Arguments:
            update {int} -- Current update
            training_stats {list} -- Statistics of the training algorithm
        """
        for key, value in training_stats.items():
            self.writer.add_scalar("training/" + key, value, update)  
    
    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        self.model.to(self.device)
        print("Model saved to " + "./models/" + self.run_id + ".nn")
    
    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.writer.close()
        except:
            pass
        
        time.sleep(1.0)
        exit(0)