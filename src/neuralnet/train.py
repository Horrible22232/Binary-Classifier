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
        --config=<path>            Path to the config file [default: ./src/neuralnet/configs/american_expr.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary and model [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    cpu = options["--cpu"]
    run_id = options["--run-id"]
    config = YamlParser(options["--config"]).get_config()

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Initialize the trainer and commence training
    trainer = Trainer(config, run_id, device)
    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()