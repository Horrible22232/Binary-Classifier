import pandas as pd
import numpy as np
import torch

class DataLoader:
    def __init__(self) -> None:
        """Initializes the data loader for the kaggle challenge.
        """
        self.dim = 186 
    
    def sample(self, num_batches: int, num_samples: int) -> dict:
        """Samples a batch of data from the data loader.
            Arguments:
                num_batches {int} -- Not used
                num_samples {int} -- The number of samples to be returned from the data
            Returns:
                {dict} -- The data to be used for training
        """
        # Load the train data and labels from the csv file
        train_data = pd.read_csv("data/train_data.csv", chunksize=500)
        train_labels = pd.read_csv("data/train_labels.csv", chunksize=num_samples)
        # Sample the first batch of data
        c_train_data = next(train_data)
        # Generate the batch in number of the sampled labels
        for num_batch, labels in enumerate(train_labels):
            # Break the loop if the number of batches is reached
            if num_batch == num_batches:
                break
            # Set the data dictionary
            data = {"samples": [], "masks": [], "labels": []}
            max_seq_len = 0
            for customer_ID, label in zip(labels["customer_ID"], labels["target"]):
                # Get the valid customers by ID
                customers = c_train_data["customer_ID"].to_numpy() == customer_ID
                # Check if the last customer is true or if the id is not in the current batch of data to load the next one
                if customers[-1] == True or customers.sum() == 0:
                    if customers[-1] == True:
                        # Concat the data from the pervious batch and the current batch to get the full sequence
                        old_c_train_data = c_train_data[-customers.sum():]
                        c_train_data = pd.concat([next(train_data), old_c_train_data])
                    else:
                        c_train_data = next(train_data)
                    
                    # Get the next valid customer IDs        
                    customers = c_train_data["customer_ID"].to_numpy() == customer_ID
                    
                # Calculate the max sequence length
                max_seq_len = max(max_seq_len, customers.sum())
                # Clean the training data by like removing NaNs, Strings, etc.
                clean_train_data = c_train_data.drop(columns=["customer_ID", "S_2"])
                clean_train_data = clean_train_data.fillna(-1.)
                clean_train_data = clean_train_data[clean_train_data.T[clean_train_data.dtypes!=np.object].index]
                customers_data = clean_train_data.to_numpy(dtype=np.float32)
                # Here you can print self.dim to set it accordingly for the model
                # print(len(customers_data[0]))
                # Select the valid customers
                customers_data = customers_data[customers]
                # Create the mask for the rnn (Many to one)
                mask = [False] * customers.sum()
                mask[-1] = True
                # Collect the data 
                data["samples"].append(customers_data)
                data["masks"].append(mask)
                data["labels"].append(label)          
            
            # Pad the sequences to the max length of the sequence
            data["samples"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["samples"]])
            data["masks"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["masks"]]).reshape(-1)
            data["labels"] = torch.tensor(data["labels"], dtype = torch.float32)
            # Return the samples, masks and labels
            yield data
            
    def sample_test(self, num_samples: int) -> dict:
        """Samples a batch of data from the data loader.
            Arguments:
                num_samples {int} -- The number of samples to be loaded from the data
            Returns:
                {dict} -- The data to be used for evaluation
        """
        # Load the test data
        test_data = pd.read_csv("data/test_data.csv", chunksize=num_samples)
        # Sample the first batch of data
        c_test_data = next(test_data)
        last_customer_id = ""
        while c_test_data is not None:
            data = {"samples": [], "masks": [], "ids": []}
            max_seq_len = 0
            for customer_ID in c_test_data["customer_ID"].unique():
                if customer_ID == last_customer_id:
                    continue
                # Don't repeat customers
                customers = c_test_data["customer_ID"].to_numpy() == customer_ID
                # Check if the last customer is true or if the id is not in the current batch of data to load the next one
                if customers[-1] == True or customers.sum() == 0:
                    if customers[-1] == True:
                        # Concat the data from the pervious batch and the current batch to get the full sequence
                        old_c_test_data = c_test_data[-customers.sum():]
                        c_test_data = pd.concat([next(test_data), old_c_test_data])
                    else:
                        c_test_data = next(test_data)
                    
                    # Get the next valid customer IDs        
                    customers = c_test_data["customer_ID"].to_numpy() == customer_ID
                    
                # Calculate the max sequence length
                max_seq_len = max(max_seq_len, customers.sum())
                # Clean the test data by like removing NaNs, Strings, etc.
                clean_test_data = c_test_data.drop(columns=["customer_ID", "S_2"])
                clean_test_data = clean_test_data.fillna(-1.)
                clean_test_data = clean_test_data[clean_test_data.T[clean_test_data.dtypes!=np.object].index]
                customers_data = clean_test_data.to_numpy(dtype=np.float32)
                # Here you can print self.dim to set it accordingly for the model
                # print(len(customers_data[0]))
                # Select the valid customers
                customers_data = customers_data[customers]
                # Create the mask for the rnn (Many to one)
                mask = [False] * customers.sum()
                mask[-1] = True
                # Collect the data 
                data["samples"].append(customers_data)
                data["masks"].append(mask)
                data["ids"].append(customer_ID)
            
                last_customer_id = customer_ID
            # Pad the sequences to the max length of the sequence
            data["samples"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["samples"]])
            data["masks"] = torch.stack([self.pad_sequence(torch.tensor(sample, dtype = torch.float32), max_seq_len) for sample in data["masks"]]).reshape(-1)
            # Return the samples, masks and customer ids
            yield data
                
    def pad_sequence(self, sequence:torch.tensor, target_length:int) -> torch.tensor:
        """Pads a sequence to the target length using zeros.
        Arguments:
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