import pandas as pd
import torch
import numpy as np

class DataLoader:
    def __init__(self) -> None:
        self.train_data = pd.read_csv("data/train_data.csv", chunksize=500)
        self.train_labels = pd.read_csv("data/train_labels.csv", chunksize=100)
        self.test_data = pd.read_csv("data/test_data.csv", chunksize=100)
        self.dim = 186 # We drop "customer_ID", "S_2"
    
    def sample(self, num_batches: int, num_samples: int) -> tuple:
        c_train_data = next(self.train_data)
        for labels in self.train_labels:
            data = {"samples": [], "masks": [], "labels": []}
            max_seq_len = 0
            for customer_ID, label in zip(labels["customer_ID"], labels["target"]):
                # Get valid customers by ID
                customers = c_train_data["customer_ID"].to_numpy() == customer_ID
                # Check if the last customer is true to load the next batch
                if customers[-1] == True or customers.sum() == 0:
                    if customers[-1] == True:
                        o_c_train_data = c_train_data[-customers.sum():]
                        c_train_data = pd.concat([next(self.train_data), o_c_train_data])
                    else:
                        # Get valid customers by ID
                        while customers.sum() == 0:
                            c_train_data = next(self.train_data)
                            customers = c_train_data["customer_ID"].to_numpy() == customer_ID
                            
                    customers = c_train_data["customer_ID"].to_numpy() == customer_ID
                    
                # Calculate max sequence length
                max_seq_len = max(max_seq_len, customers.sum())
                # Clean training data
                clean_train_data = c_train_data.drop(columns=["customer_ID", "S_2"])
                clean_train_data = clean_train_data.fillna(-1.)
                clean_train_data = clean_train_data[clean_train_data.T[clean_train_data.dtypes!=np.object].index]
                customers_data = clean_train_data.to_numpy(dtype=np.float32)
                # self.dim
                # print(len(customers_data[0]))
                # Select valid customers
                customers_data = customers_data[customers]
                # Create mask for the rnn
                mask = [False] * customers.sum()
                mask[-1] = True
                # Collect the data 
                data["samples"].append(customers_data)
                data["masks"].append(mask)
                data["labels"].append(label)
                
            
            # Pad the sequences
            data["samples"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["samples"]])
            data["masks"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["masks"]])
            data["masks"] = data["masks"].reshape(-1)
            data["labels"] = torch.tensor(data["labels"], dtype = torch.float32)
            yield data
                
    def pad_sequence(self, sequence:torch.tensor, target_length:int) -> torch.tensor:
        """Pads a sequence to the target length using zeros.
        Args:
            sequence {torch.tensor} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence
        Returns:
            {torch.tensor} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array (e.g. visual observation)
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)