import numpy
import copy_reg
from multiprocessing import RawArray
import ctypes
import mmap
__all__ = ['empty', 'empty_like', 'copy']

def empty_like(array, dtype=None):
  if dtype is None: dtype = array.dtype
  return anonymousmemmap(numpy.broadcast(array, array).shape, dtype)

def empty(shape, dtype='f8'):
  """ allocates an empty array on the shared memory """
  return anonymousmemmap(shape, dtype)

def copy(a):
  """ copies an array to the shared memory, use
     a = copy(a) to immediately dereference the old 'a' on private memory
   """
  shared = anonymousmemmap(a.shape, dtype=a.dtype)
  shared[:] = a[:]
  return shared

def fromiter(iter, dtype, count=None):
    return copy(numpy.fromiter(iter, dtype, count))

def __unpickle__(ai, dtype):
  dtype = numpy.dtype(dtype)
  tp = numpy.ctypeslib._typecodes['|u1']
  # if there are strides, use strides, otherwise the stride is the itemsize of dtype
  if ai['strides']:
    tp *= ai['strides'][-1]
  else:
    tp *= dtype.itemsize
  for i in numpy.asarray(ai['shape'])[::-1]:
    tp *= i
  # grab a flat char array at the sharemem address, with length at least contain ai required
  ra = tp.from_address(ai['data'][0])
  buffer = numpy.ctypeslib.as_array(ra).ravel()
  # view it as what it should look like
  shm = numpy.ndarray(buffer=buffer, dtype=dtype, 
      strides=ai['strides'], shape=ai['shape']).view(type=anonymousmemmap)
  return shm

def __pickle__(obj):
  return obj.__reduce__()

class anonymousmemmap(numpy.memmap):
    def __new__(subtype, shape, dtype=numpy.uint8, order='C'):

        descr = numpy.dtype(dtype)
        _dbytes = descr.itemsize

        shape = numpy.atleast_1d(shape)
        size = 1
        for k in shape:
            size *= k

        bytes = long(size*_dbytes)

        mm = mmap.mmap(-1, bytes)

        self = numpy.ndarray.__new__(subtype, shape, dtype=descr, buffer=mm, order=order)
        self._mmap = mm
        return self

    def __array_wrap__(self, outarr, context=None):
    # after ufunc this won't be on shm!
        return numpy.ndarray.__array_wrap__(self.view(numpy.ndarray), outarr, context)

    def __reduce__(self):
        if hasattr(self, '_mmap'):
            return __unpickle__, (self.__array_interface__, self.dtype)
        else:
            return numpy.ndarray.__reduce__(self)

copy_reg.pickle(anonymousmemmap, __pickle__, __unpickle__)

