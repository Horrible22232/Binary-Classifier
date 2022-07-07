import numpy as np

class SinusGenerator: 
    """A generator that samples a sinus- and a cosinus function between 0 and 1."""
    def __init__(self) -> None:
        """Initializes a sinus generator. 
        """
        self.cos = lambda x: np.cos(x)
        self.sin = lambda x: np.sin(x)
        self.func = [self.cos, self.sin]
        self.dim = 2
    
    def sample(self, num_batches: int, num_samples: int) -> tuple:
        """Generates a batch of samples.
        Arguments:
            num_batches {int} -- number of batches to generate
            num_samples {int} -- number of samples per batch  
        Returns:
            {tuple} -- (data, label) Data describes the x-value and fuction value. The label indicates if the function is sinus or cosinus.
                If the label is 1, then the sinus funciton will be sampled at x_val else the cosinus function will be sampled.
        """
        for _ in range(num_batches):
            # Generate random x-values between 0 and 1
            x_vals = np.random.uniform(low = 0.0, high=1.0, size=num_samples)
            # Generate random 0 or 1 labels
            labels = np.random.randint(2, size=num_samples).astype(np.int).tolist()
            # Evaluate the function at the x-values
            func_vals = [self.func[label](x) for x, label in zip(x_vals, labels)]
            # Create the samples
            samples = [[x_val, func_val] for x_val, func_val in zip(x_vals, func_vals)]
            # Create the dictionary to be returned
            data = {"samples": samples, "label": labels}
            # Return the data and labels
            yield data
            
class NegativeSeqDataGen():
    """Genates a sequence of random values between 0 and 1 with one negative value in the middle of the sequence.
        The label is 1 if the negative value is in the middle of the sequence and 0 otherwise."""
    def __init__(self, config) -> None:
        self.sequence_len = config["sequence_len"]
        self.dim = 1
        
    def sample(self, num_batches: int, num_samples: int) -> tuple:
        """Generates a batch of samples.
        Arguments:
            num_batches {int} -- number of batches to generate
            num_samples {int} -- number of samples per batch  
        Returns:
            {tuple} -- (data, label, mask) The data to be used for training
        """
        for _ in range(num_batches):
            # Generate random values between 0 and 1
            samples = np.random.random((num_samples, self.sequence_len))
            # Generate random 0 or 1 labels
            labels = np.random.randint(2, size=num_samples).astype(np.int).tolist()
            # Generate a mask
            mask = np.zeros((num_samples, self.sequence_len))
            mask[:, -1] = 1
            mask = mask.flatten()
            # Set for a sample with label 1 a random value negative 
            for i in range(num_samples):
                if labels[i] == 1:
                    rand_idx = np.random.randint(self.sequence_len)
                    samples[i][rand_idx] *= -1
            # Create the proper sequence data structure
            samples = np.expand_dims(samples, axis=1)
            samples = samples.swapaxes(1, -1)
            # Create the dictionary to be returned
            data = {"samples": samples, "label": labels, "mask": mask}
            # Return the data
            yield data