import compute as np
from timeit import timeit

print(timeit(lambda: np.arange(2000000), number=10))

import numpy

print(timeit(lambda: numpy.arange(2000000), number=10))


t = np.arange(2000000)

t / (t + 0.000001)


print(timeit(lambda: t ** 1.5, number=10))

a = t.array

print(timeit(lambda: a ** 1.5, number=10))
