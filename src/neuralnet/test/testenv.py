import numpy as np

class SinEnv: 
    """An Environment that samples a sinus function and a cosinus function."""
    def __init__(self, n: int) -> None:
        """ Initializes a Sinus Environment. 

        Arguments:
            n {int} -- number of samples to generate
        """
        self.n = n
        self.cos = lambda x: np.cos(x)
        self.sin = lambda x: np.sin(x)
        self.func = [self.cos, self.sin]
    
    def sample(self, num_samples: int) -> tuple:
        """ Samples the environment.
        Arguments:
        {int} num_samples -- number of samples to generate
        
        Returns:
        {tuple} -- (data, label) Data describes the x-value and fuction value. The label indicates if the function is sinus or cosinus.
        If the label is 1, then the sinus funciton will be sampled at x_val else the cosinus function will be sampled.
        """
        for _ in range(num_samples):
            x_vals = np.random.uniform(low = 0.0, high=1.0, size=self.n)
            labels = np.random.randint(2, size=self.n).astype(np.int).tolist()
            func_vals = [self.func[label](x) for x, label in zip(x_vals, labels)]
            data = [[x_val, func_val] for x_val, func_val in zip(x_vals, func_vals)]
            yield data, labels
        