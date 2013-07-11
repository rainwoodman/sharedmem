import sharedmem

import numpy

a = numpy.random.random(10000)
arg = sharedmem.argsort(a, chunksize=1024)

print a[arg]
assert (a[arg[1:]] >= a[arg[:-1]]).all()

