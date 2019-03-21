import numpy as np


def reward(arr):
    if len(arr) > 1:
        r = -(arr[0, -1] - arr[0, -2])
    else:
        r = -arr[0]

    return r



def rms(arr):

    return np.sqrt(np.sum(np.power(arr, 2)))
