import pickle
import torch
import numpy as np
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
        --model=<path>              Specifies the path to the trained model [default: ./src/neuralnet/models/run.nn].
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
    
    