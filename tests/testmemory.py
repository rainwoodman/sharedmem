import sharedmem
import pickle
import numpy
a = sharedmem.empty(100)
a[:] = range(100)
s = pickle.dumps(a)
b = pickle.loads(s)

assert isinstance(b, type(a))
assert (a == b).all()

b[:] += 10
assert isinstance(b, type(a))
assert (a == b).all()

assert not isinstance(a + 10, type(a))
assert not isinstance(numpy.sum(a), type(a))
assert not isinstance(a + b, type(a))
assert not isinstance(a * b, type(a))
