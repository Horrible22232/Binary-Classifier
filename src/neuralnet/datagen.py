import numpy as np

class SinusGenerator: 
    """A generator that samples a sinus- and a cosinus function between 0 and 1."""
    def __init__(self) -> None:
        """Initializes a sinus generator. 
        """
        self.cos = lambda x: np.cos(x)
        self.sin = lambda x: np.sin(x)
        self.func = [self.cos, self.sin]
    
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
        