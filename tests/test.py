import numpy as np


def data_generator():
    dataset = np.array(range(5))
    for d in dataset:
        yield d

for n in data_generator():
    print(n)
