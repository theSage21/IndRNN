from indrnn import IndRNN
import numpy as np
import random
import torch


def addition(length, batchsize):
    seq = np.random.uniform(0, 1, length)
    ind = np.random.choice(range(length), 2, replace=False)
    markers = np.zeros(length)
    markers[ind] = 1


for batch in addition(100, 32):
