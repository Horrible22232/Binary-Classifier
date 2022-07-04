import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, in_features, config: dict) -> None:
        """Initializes the classifier model.

        Args:
            in_features {int}: The number of features in the input data.
            config {dict}: The model configuration.
        """
        self.linear1 = nn.Linear(in_features, config['hidden_size'])
        self.activ_fn = nn.ReLU()
        self.linear2 = nn.Linear(config['hidden_size'], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the model.

        Arguments:
            data {torch.Tensor} -- The input data for the model.

        Returns:
            {torch.Tensor} -- The output of the model, wether the data is a positive or negative example.
        """
        h = self.activ_fn(self.linear1(data))
        h = self.linear2(h)
        h = self.sigmoid(h)
        return h