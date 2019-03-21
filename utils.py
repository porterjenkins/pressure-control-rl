import numpy as np


def reward(arr):
    r = -(arr[0, -1] - arr[0, -2])

    return r



def rms(arr):

    return np.sqrt(np.sum(np.power(arr, 2)))
