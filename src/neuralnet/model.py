from turtle import forward
import torch
from torch import nn

class RecurrentVecEncoder(nn.Module):
    def __init__(self, in_features, config: dict, device:torch.device) -> None:
        """Initializes a recurrent vector encoder.
            Arguments:
                in_features {int} -- The number of features in the input data.
                config {dict} -- The model configuration.
                device {torch.device} -- The device to use for the model.
        """
        super().__init__()
        self.device = device
        self.config = config
        self.vec_encoder = nn.Linear(in_features, self.config['hidden_size'])
        self.activ_fn = nn.ReLU()
        if self.config["encoder"] == "lstm":
            self.recurrent_layer = nn.LSTM(self.config["hidden_size"], self.config["hidden_state_size"], batch_first=True)
        else:
            self.recurrent_layer = nn.GRU(self.config['hidden_size'], self.config['hidden_state_size'], batch_first=True)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Encodes the input vector.
        Arguments:
            data {torch.Tensor} -- input sequence 
        Returns:
            {torch.Tensor} -- encoded sequence
        """
        # Get the number of sequences
        num_sequences, sequence_len = h.shape[0], h.shape[1]
        # Flatten the input vector
        h = h.view(num_sequences * sequence_len, -1)
        # Encode the input vector
        h = self.activ_fn(self.vec_encoder(h))
        # Reshape the encoded vector to the original shape
        h = h.view(num_sequences, sequence_len, -1)
        # Initialize the recurrent cell states
        memory = self.init_recurrent_cell_states(num_sequences)
        # Forward recurrent layer
        h, memory = self.recurrent_layer(h, memory)
        
        return h
    
    def init_recurrent_cell_states(self, num_sequences:int) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.
        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros((num_sequences), self.config["hidden_state_size"], dtype=torch.float32, device=self.device).unsqueeze(0)
        cxs = None
        if self.config["encoder"] == "lstm":
            cxs = torch.zeros((num_sequences), self.config["hidden_state_size"], dtype=torch.float32, device=self.device).unsqueeze(0)
            return hxs, cxs
        return hxs

class Classifier(nn.Module):
    def __init__(self, in_features, config: dict, device:torch.device) -> None:
        """Initializes the classifier model.
        Arguments:
            in_features {int} -- The number of features in the input data.
            config {dict} -- The model configuration.
            device {torch.device} -- The device to use for the model.
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