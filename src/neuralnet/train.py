import torch
from docopt import docopt
from trainer import Trainer
from yaml_parser import YamlParser

def main() -> None:
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the config file [default: ./configs/default.yaml]
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    cpu = options["--cpu"]
    config = YamlParser(options["--config"]).get_config()

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # Initialize the trainer and commence training
    trainer = Trainer(config, device=device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()