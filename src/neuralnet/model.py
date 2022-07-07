from turtle import forward
import torch
from torch import nn

class RecurrentVecEncoder(nn.Module):
    def __init__(self) -> None:
        """Initializes a recurrent vector encoder.
        """
        super().__init__()
        
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Encodes the input vector.
        Arguments:
            {torch.Tensor} data -- input sequence
        Returns:
            {torch.Tensor} -- encoded sequence
        """
        return data

class Classifier(nn.Module):
    def __init__(self, in_features, config: dict, device) -> None:
        """Initializes the classifier model.

        Arguments:
            in_features {int} -- The number of features in the input data.
            config {dict} -- The model configuration.
        """
        super().__init__()
        self.encoder = nn.Linear(in_features, config['hidden_size'])
        self.activ_fn = nn.ReLU()
        self.out = nn.Linear(config['hidden_size'], 1)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Arguments:
            data {torch.Tensor} -- The input data for the model.

        Returns:
            {torch.Tensor} -- The output of the model, wether the data is a positive or negative example.
        """
        h = self.activ_fn(self.encoder(data))
        h = self.out(h).squeeze()
        return h