import compute as np
from timeit import timeit

print(timeit(lambda: np.arange(2000000), number=10))

import numpy

print(timeit(lambda: numpy.arange(2000000), number=10))

t = np.arange(20000)
