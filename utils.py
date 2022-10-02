from dataloader import DataLoader
from datagen import SinusGenerator, NegativeSeqGen

def create_data_loader(config:dict):
    """Initializes the data loader based on the provided name
    Arguments:
        config {dict} -- Configuration of the data loader
    Returns:
        {data_loader} -- The initialized data loader
    """
    if config["name"] == "SinusGenerator":
        return SinusGenerator()
    if config["name"] == "NegativeSeqGen":
        return NegativeSeqGen(config["sequence_len"])
    if config["name"] == "AmericanExpr":
        return DataLoader()