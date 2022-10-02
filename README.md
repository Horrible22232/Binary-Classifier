# Overview
This repository provides the code of a binary classifier that is well documented, tested and was part of the [Kaggle: American Express - Default Prediction challenge](https://www.kaggle.com/competitions/amex-default-prediction/). The classifier scored rank 4086 due to its simplicity. 

# Contributions
The contributions are as follows:
* Binary classfier
* Optional recurrent encoder => Can be used for sequential data

# Tests
The classifier got tested under two artificial problems:
* In the first problem, the classifier should determine whether a sample point x = (x0, x1) belongs to the sinus or cosinus function:

|  x0: x  |  x1: f(x)  |  label  |
|---|---|---|
|  0  |  0  |  1  |
|  0.4  |  0.39  |  1  |
|  0.4  |  0.92  |  0  |

* In the second problem, the classifier gets sequential data as input and should determine whether a negative value is part of the sequence x:

|  x00  |  x10  |  x20  |  x30  |  label  |
|---|---|---|---|---|
|  0.24  |  0.43  |  0.3  | 0.11  |  0  |
|  0.21 |  -0.11  |  0.94  |  0.56  |  1  |
|  0.67  |  0.35  |  0.31  |  0.22  |  0  |

# American Express Challenge
In the American Express challenge, the data couldn't be fed into the model, so there was some preprocessing of it required. The data consists out of 186 features in total, where some of them were NaN values or not normalized. Furthermore, there was a lot of correlation and some of the features were categorical. To make the model work, just the NaN values got heuristically replaced with minus one. The other issues with the data got ignored. In the next step, the data got divided into sequences. However, the sequence length wasn't consisted for each data point, so there was zero padding necessary. In the end, the padded values got masked to not interfere the classification process. This problem which had to be solved was a many-to-one problem. The used model consisted out of one preprocessing layer to shape the input adequately for the four layer recurrent gru encoder with a hidden size of 512. One more final layer was used for the classification problem. 

# Try it out!
It is recommended to try the code out by yourself. Follow these steps and then you are good to go:
1. Install PyTorch
2. Install the necessary requirements: `pip install -r requirements.txt`
3. Try it out: `python train.py`



