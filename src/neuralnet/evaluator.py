import csv
import pickle
import torch
from docopt import docopt
from model import Classifier
from dataloader import DataLoader
from utils import create_data_loader

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        evaluator.py [options]
        evaluator.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/run.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]
    device = torch.device("cpu")
    
    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))
    
    # Instantiate the data loader
    data_loader = create_data_loader(config["data"])
    
    # Initialize model and load its parameters
    model = Classifier(data_loader.dim, config["model"])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create th file writer for the submission file
    file = open('submission.csv','w')
    # Add the header to the file
    file.write("customer_ID,prediction")
    # Classify the test data
    num_samples = 0
    for data in data_loader.sample_test(num_samples=100):
        # Get the samples and labels
        samples, ids = data["samples"], data["ids"]
        # Get the output of the model
        output = model(samples)
        # Mask the output if necessary
        if "masks" in data:
            output = output[data["masks"] == 1]
        # Get the prediction probability
        output = torch.sigmoid(output).detach().cpu().numpy()
        # Round the probabilities
        output = output.round(decimals=2).astype('str')
        # Write the results to the file
        txt = '\n'
        for user in list(zip(ids, output)):
            txt += ",".join(user) + '\n'
        file.write(txt[:-2])
        # Print the progress
        num_samples += len(ids)
        print("Classified {} samples".format(num_samples))
        break
    file.close()

if __name__ == "__main__":
    main()    