from ruamel.yaml import YAML

class YamlParser:
    """The YamlParser parses a yaml file containing parameters for the model and trainer.
    The data is parsed during initialization.
    Retrieve the parameters using the get_config function.
    The data can be accessed like:
    parser.get_config()["model"]["name"]
    """

    def __init__(self, path):
        """Loads and prepares the specified config file.
        Arguments:
            path {str} -- Yaml file path to the to be loaded config file.
        """
        # Load the config file
        stream = open(path, "r")
        yaml = YAML()
        yaml_args = yaml.load_all(stream)
        
        # Final contents of the config file will be added to a dictionary
        self._config = {}

        # Prepare data
        for data in yaml_args:
            self._config = dict(data)

    def get_config(self):
        """ 
        Returns:
            {dict} -- Nested dictionary that contains configs for the  model and trainer.
        """
        return self._config