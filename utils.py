import numpy as np


def reward(arr):
    if len(arr) > 1:
        r = -(arr[-1] - arr[-2])
    else:
        r = -arr[0]

    return r



def rms(arr):

    return np.sqrt(np.sum(np.power(arr, 2)))
