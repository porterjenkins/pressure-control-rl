import numpy as np


def reward(prms, target):
    r = -np.abs(prms - target) / 1.0e2

    return r



def rms(arr):

    return np.sqrt(np.mean(np.power(arr, 2)))
