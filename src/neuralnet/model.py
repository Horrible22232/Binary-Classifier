import torch
from torch import nn

class RecurrentVecEncoder(nn.Module):
    def __init__(self, in_features, config: dict) -> None:
        """Initializes a recurrent vector encoder.
            Arguments:
                in_features {int} -- The number of features in the input data.
                config {dict} -- The model configuration.
                device {torch.device} -- The device to use for the model.
        """
        super().__init__()
        self.config = config
        self.vec_encoder = nn.Linear(in_features, self.config['hidden_size'])
        self.activ_fn = nn.ReLU()
        if self.config["rnn"] == "lstm":
            self.recurrent_layer = nn.LSTM(self.config["hidden_size"], self.config["hidden_state_size"], num_layers=self.config["num_layers"], batch_first=True)
        else:
            self.recurrent_layer = nn.GRU(self.config['hidden_size'], self.config['hidden_state_size'], num_layers=self.config["num_layers"], batch_first=True)
        
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
        h = h.reshape(num_sequences * sequence_len, -1)
        # Encode the input vector
        h = self.activ_fn(self.vec_encoder(h))
        # Reshape the encoded vector to the original shape
        h = h.reshape(num_sequences, sequence_len, -1)
        # Forward recurrent layer
        h, _ = self.recurrent_layer(h, None)
        # Flatten the output vector
        h = h.reshape(num_sequences * sequence_len, -1)
        return h

class Classifier(nn.Module):
    def __init__(self, in_features, config: dict) -> None:
        """Initializes the classifier model.
        Arguments:
            in_features {int} -- The number of features in the input data.
            config {dict} -- The model configuration.
        """
        super().__init__()
        if config["encoder"] == "VecEncoder":
            self.encoder = nn.Linear(in_features, config['hidden_size'])
            in_features_next_layer  = config['hidden_size']
        else:
            self.encoder = RecurrentVecEncoder(in_features, config)
            in_features_next_layer  = config['hidden_state_size']
        self.activ_fn = nn.ReLU()
        self.out = nn.Linear(in_features_next_layer, 1)
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """The forward pass of the model.
        Arguments:
            data {torch.Tensor} -- The input data for the model.
        Returns:
            {torch.Tensor} -- The output of the model, wether the data is a positive or negative example.
        """
        h = self.activ_fn(self.encoder(data))
        h = self.out(h).squeeze()
        return h