import numpy
from .. import core as sharedmem

__all__ = ['fromfile', 'tofile', 'take']

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
  with sharedmem.TPool(np=np) as pool:
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
  with Sharedmem.TPool(np=np) as pool:
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
  if sharedmem.cpu_count() <= 1:
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

  with sharedmem.TPool() as pool:
    chunksize = len(indices) // pool.np
    if chunksize < 1: chunksize = 1
    def work(i):
      #needs numpy ticket #2156
      sl = slice(i, i+chunksize)
      if len(out) == 0: return
      source.take(indices[sl], axis=axis, out=out[sl], mode=mode)
    pool.map(work, range(0, len(indices), chunksize))
  return out

