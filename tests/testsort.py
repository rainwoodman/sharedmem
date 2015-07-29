raise NotImplemented("sorting has been deprecated")

import sharedmem

import numpy
numpy.random.seed(1)
a = numpy.random.random(10000000)
arg = sharedmem.argsort(a, chunksize=10240)

print a[arg]
assert (a[arg[1:]] >= a[arg[:-1]]).all()

