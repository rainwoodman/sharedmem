import numpy
import copy_reg
from multiprocessing import RawArray
import ctypes

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

def fromfile(filename, dtype, count=None, chunksize=1024 * 1024 * 64, np=None):
  """ the default size 64MB agrees with lustre block size but is not an optimized choice.
  """
  dtype = numpy.dtype(dtype)
  if hasattr(filename, 'seek'):
    file = filename
  else:
    file = open(filename)

  cur = file.tell()

  if count is None:
    file.seek(0, os.SEEK_END)
    length = file.tell()
    count = (length - cur) / dtype.itemsize
    file.seek(cur, os.SEEK_SET)

  buffer = numpy.empty(dtype=dtype, shape=count) 
  start = numpy.arange(0, count, chunksize)
  stop = start + chunksize
  with Pool(use_threads=True, np=np) as pool:
    def work(start, stop):
      if not hasattr(pool.local, 'file'):
        pool.local.file = open(file.name)
      start, stop, step = slice(start, stop).indices(count)
      pool.local.file.seek(cur + start * dtype.itemsize, os.SEEK_SET)
      buffer[start:stop] = numpy.fromfile(pool.local.file, count=stop-start, dtype=dtype)
    pool.starmap(work, zip(start, stop))

  file.seek(cur + count * dtype.itemsize, os.SEEK_SET)
  return buffer

def tofile(file, array, np=None):
  """ write an array to file in parallel with mmap"""
  file = numpy.memmap(file, array.dtype, 'w+', shape=array.shape)
  with Pool(use_threads=True, np=np) as pool:
    chunksize = len(array) // np
    if chunksize < 1: chunksize = 1
    def writechunk(i):
      file[i:i+chunksize]= array[i:i+chunksize]
    pool.map(writechunk, range(0, len(array), chunksize))
  file.flush()

def take(source, indices, axis=None, out=None, mode='wrap'):
  """ the default mode is 'wrap', because 'raise' will copy the output array.
      we use threads because it is faster than processes.
      need the patch on numpy ticket 2131 to release the GIL.

  """
  if cpu_count() <= 1:
    return numpy.take(source, indices, axis, out, mode)

  indices = numpy.asarray(indices, dtype='i8')
  if out is None:
    if axis is None:
      out = numpy.empty(dtype=source.dtype, shape=indices.shape)
    else:
      shape = []
      for d, n in enumerate(source.shape):
        if d < axis or d > axis:
          shape += [n]
        else:
          for dd, nn in enumerate(indices.shape):
            shape += [nn]
      out = numpy.empty(dtype=source.dtype, shape=shape)

  with Pool(use_threads=True) as pool:
    chunksize = len(indices) // pool.np
    if chunksize < 1: chunksize = 1
    def work(i):
      #needs numpy ticket #2156
      sl = slice(i, i+chunksize)
      if len(out) == 0: return
      source.take(indices[sl], axis=axis, out=out[sl], mode=mode)
    pool.map(work, range(0, len(indices), chunksize))
  return out

