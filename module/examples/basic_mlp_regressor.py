import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import KFold
import torch


from data import *
from model import *
from setup import *
from run import *


def main():
    """
    Basic example of using sklearn to train and test model
    """

    model = get_mlp_regressor_sklearn(learning_rate_init=0.01, max_iter=1000)

    train_x = torch.rand(40, 1)
    train_y = (train_x * 3 + 2).squeeze()

    print(f"{train_x.squeeze()}")
    print(f"{train_y}\n")

    final_loss = run_train_sklearn(model, train_x, train_y)

    print(f"Final Accuracy: {final_loss}\n")

    test_x = torch.rand(40, 1)
    test_y = (train_x * 3 + 2).squeeze()

    print(f"{test_x.squeeze()}")
    print(f"{test_y}")

    test_y_hat = run_test_sklearn(model, test_x)

    print(f"{test_y_hat}\n")

    pass


if __name__ == "__main__":
    main()
