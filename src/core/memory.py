import numpy
import copy_reg
from multiprocessing import RawArray
import ctypes

__all__ = ['empty', 'empty_like', 'copy']

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
      strides=ai['strides'], shape=ai['shape']).view(type=SharedMemArray)
  return shm

def __pickle__(obj):
  return obj.__reduce__()

class SharedMemArray(numpy.ndarray):
  """ 
      SharedMemArray works with multiprocessing.Pool through pickling.
      With sharedmem.Pool pickling is unnecssary. sharemem.Pool is recommended.

      Do not directly create an SharedMemArray or pass it to numpy.view.
      Use sharedmem.empty or sharedmem.copy instead.

      When a SharedMemArray is pickled, only the meta information is stored,
      So that when it is unpicled on the other process, the data is not copied,
      but simply viewed on the same address.
  """
  __array_priority__ = -900.0

  def __new__(cls, shape, dtype='f8'):
    dtype = numpy.dtype(dtype)
    tp = ctypes.c_byte * dtype.itemsize
    ra = RawArray(tp, int(numpy.asarray(shape).prod()))
    shm = numpy.ctypeslib.as_array(ra)
    if not shape:
      fullshape = dtype.shape
    else:
      if not dtype.shape:
        fullshape = shape
      else:
        if not hasattr(shape, "__iter__"):
          shape = [shape]
        else:
          shape = list(shape)
        if not hasattr(dtype.shape, "__iter__"):
          dshape += [dtype.shape]
        else:
          dshape = list(dtype.shape)
        fullshape = shape + dshape
    return shm.view(dtype=dtype.base, type=SharedMemArray).reshape(fullshape)
    
  def __reduce__(self):
    return __unpickle__, (self.__array_interface__, self.dtype)

copy_reg.pickle(SharedMemArray, __pickle__, __unpickle__)

def empty_like(array, dtype=None):
  if dtype is None: dtype = array.dtype
  return SharedMemArray(array.shape, dtype)

def empty(shape, dtype='f8'):
  """ allocates an empty array on the shared memory """
  return SharedMemArray(shape, dtype)

def copy(a):
  """ copies an array to the shared memory, use
     a = copy(a) to immediately dereference the old 'a' on private memory
   """
  shared = SharedMemArray(a.shape, dtype=a.dtype)
  shared[:] = a[:]
  return shared
