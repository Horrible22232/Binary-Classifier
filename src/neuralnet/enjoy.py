import pickle
import torch
import numpy as np
import pandas as pd
from docopt import docopt
from model import Classifier
from utils import create_data_loader

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]
    device = torch.device("cpu")
    
    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))
    
    # Instantiate the data loader
    data_loader = create_data_loader(config)
    
    # Initialize model and load its parameters
    model = Classifier(2, config, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Generate data
    data, label = list(data_loader.sample(num_batches=1, num_samples=10))[0]
    # Convert the data to a tensor
    data = torch.tensor(data, dtype=torch.float32)
    # Get the output of the model
    output = model(data)
    # Get the prediction probability
    output = torch.sigmoid(output).detach()
    # Get the predicted label
    pred_label = torch.bernoulli(output).int()
    # Create a dataframe with the data and the predicted label
    data_dict = {"data_{}".format(i): data[:, i] for i in range(data.shape[1])}
    results = pd.DataFrame.from_dict({**data_dict, "probability": np.around(output, decimals=4), "pred_label": pred_label, "label": label})
    
    print(results)
    
if __name__ == "__main__":
    main()  
    
    