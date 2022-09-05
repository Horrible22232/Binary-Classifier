# Overview
This repository provides the code of a binary classifier that is well documented, tested and was part of the [Kaggle: American Express - Default Prediction challenge](https://www.kaggle.com/competitions/amex-default-prediction/). The classifier scored rank 4086 due to its simplicity. 

# Contributions
The contributions are as follows:
* Binary classfier
* Optional recurrent encoder => Can be used for sequential data

# Tests
The classifier got tested under two artificial problems:
* In the first problem, the classifier should determine wether a sample points belongs to the sinus or cosinus function:

|  x0: x  |  x1: f(x)  |  label  |
|---|---|---|
|  0  |  0  |  1  |
|  0.4  |  0.39  |  1  |
|  0.4  |  0.92  |  0  |

* In the second problem, the classifier gets sequential data as input and should determine wether a negative value is part of the sequence:

|  x00  |  x10  |  x20  |  x30  |  label  |
|---|---|---|---|---|
|  0.24  |  0.43  |  0.3  | 0.11  |  0  |
|  0.21 |  -0.11  |  0.94  |  0.56  |  1  |
|  0.67  |  0.35  |  0.31  |  0.22  |  0  |

# American Express Challenge
* Used model

# Future Work

# Try it out!




