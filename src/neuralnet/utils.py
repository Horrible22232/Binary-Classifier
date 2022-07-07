from datagen import SinusGenerator, NegativeSeqDataGen

def create_data_loader(config:dict):
    """Initializes the data loader based on the provided name
    Arguments:
        config {dict} -- Configuration of the data loader
    Returns:
        {data_loader} -- The initialized data loader
    """
    if config["name"] == "SinusGenerator":
        return SinusGenerator()
    if config["name"] == "NegativeSeqDataGen":
        return NegativeSeqDataGen(config["sequence_len"])