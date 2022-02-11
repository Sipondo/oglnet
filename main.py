import compute as np
from timeit import timeit

print(timeit(lambda: np.arange(20000000), number=10))

import numpy

print(timeit(lambda: numpy.arange(20000000), number=10))
