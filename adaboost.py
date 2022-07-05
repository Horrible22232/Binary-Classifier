import numpy as np

def load_models() -> None:
    """
    Loads the models from the disk
    
    Returns:
        {list} -- List of models
    """
    model_0 = lambda x: True
    model_1 = lambda x: False
    model_2 = lambda x: True
    
    return [model_0, model_1, model_2]

# Init
models = load_models()
data = [1, 2, 3]

# Classify the data
for x in data:
    evals = []
    for model in models:
        evals.append(model(x))

    print(np.round(np.mean(evals) * 100, 2),"% of the models voted for yes", sep="")