import compute as np
from timeit import timeit

print(timeit(lambda: np.arange(2000000), number=10))

import numpy

print(timeit(lambda: numpy.arange(2000000), number=10))


t = np.arange(20000)


t.reshape(1000, 20)

t.shape

np.arange(10) * 3

np.arange(983475) / 983475
