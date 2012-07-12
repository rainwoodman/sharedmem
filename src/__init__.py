"""
sharedmem facilities SHM parallization.
empty and wrap allocates numpy arrays on the SHM
Pool is a slave-pool that can be either based on Threads or Processes.

Notice that Pool.map and Pool.star map do not return ordered results.

"""

import multiprocessing as mp
import numpy
import os
import threading
import Queue as queue
import ctypes
import traceback
import copy_reg
import signal
import itertools

from numpy import ctypeslib
from multiprocessing.sharedctypes import RawArray
from listtools import cycle, zip, repeat
from warnings import warn
import heapq

from gaepsi.tools import array_split

__shmdebug__ = False
__timeout__ = 10

def set_debug(flag):
  """ in debug mode (flag==True), no slaves are spawn,
      rather all work are done in serial on the master thread/process.
      so that if the worker throws out exceptions, debugging from the main
      process context is possible. (in iptyhon, with debug magic command, eg)
  """
  global __shmdebug__
  __shmdebug__ = flag

def set_timeout(timeout):
  """ set max number of timeout retries. each retry we wait for 10 secs. 
      if the master fails to join all slaves after timeout retries,
      a warning will be issued with the total number of alive slaves
      reported. 
      The program continues, but there may be a serious issue in the worker 
      functions.
  """
  global __timeout__
  ret = timeout
  __timeout__ = timeout
  return ret

def cpu_count():
  """ The cpu count defaults to the number of physical cpu cores
      but can be set with OMP_NUM_THREADS environment variable.
      OMP_NUM_THREADS is used because if you hybrid sharedmem with
      some openMP extenstions one environment will do it all.

      on some machines the physical number of cores does not equal
      the number of cpus shall be used. PSC Blacklight for example.

      Pool defaults to use cpu_count() slaves. however it can be overridden
      in Pool.
  """
  num = os.getenv("OMP_NUM_THREADS")
  try:
    return int(num)
  except:
    return mp.cpu_count()

class Pool:
  """
    with Pool() as p
      def work(a, b, c):
        pass
      p.starmap(work, zip(A, B, C))

    To use a Thread pool, pass use_threads=True
    there is a Lock accessible as 'with p.lock'
    
  """
  def __enter__(self):
    return self
  def __exit__(self, type, value, traceback):
    pass
  @property
  def rank(self):
    return self._local._rank

  @property 
  def local(self):
    return self._local

  def __init__(self, np=None, use_threads=False):
    if np is None: np = cpu_count()
    self.np = np
    self._local = None
    if use_threads:
      self.QueueFactory = queue.Queue
      self.JoinableQueueFactory = queue.Queue
      def func(*args, **kwargs):
        slave = threading.Thread(*args, **kwargs)
        slave.daemon = True
        return slave
      self.SlaveFactory = func
      self.lock = threading.Lock()
      self._local = threading.local()
    else:
      self.QueueFactory = mp.Queue
      self.JoinableQueueFactory = mp.JoinableQueue
      def func(*args, **kwargs):
        slave = mp.Process(*args, **kwargs)
        slave.daemon = True
        return slave
      self.SlaveFactory = func
      self.lock = mp.Lock()
      self._local = lambda: None
     # master threads's rank is None
      self._local.rank = None

  def zipsplit(self, list, nchunks=None, chunksize=None):
    return zip(*self.split(list, nchunks, chunksize))

  def split(self, list, nchunks=None, chunksize=None):
    """ Split every item in the list into nchunks, and return a list of chunked items.
           - then used with p.starmap(work, zip(*p.split((xxx,xxx,xxxx), chunksize=1024))
        For non sequence items, tuples, and 0d arrays, constructs a repeated iterator,
        For sequence items(but tuples), convert to numpy array then use array_split to split them.
        either give nchunks or chunksize. chunksize is only instructive, nchunk is estimated from chunksize
    """
    result = []
    if nchunks is None:
      if chunksize is None:
        nchunks = self.np * 2
      else:
        nchunks = 0
        for item in list:
          if hasattr(item, '__len__') and not isinstance(item, tuple):
            nchunks = int(len(item) / chunksize)
        if nchunks == 0: nchunks = 1
      
    for item in list:
      if isinstance(item, tuple):
        result += [repeat(item)]
      else:
        aitem = numpy.asarray(item)
        if aitem.shape:
          result += [array_split(aitem, nchunks)]
        else:
          result += [repeat(item)]

    return result

  def starmap(self, work, sequence, chunksize=1, ordered=False):
    return self.map(work, sequence, chunksize, ordered=ordered, star=True)

  def do(self, jobs):
    def work(job):
      job()
    return self.map(work, jobs, chunksize=1, ordered=False, star=False)

  def map(self, work, sequence, chunksize=1, ordered=False, star=False):
    """
      calls work on every item in sequence. the return value is unordered unless ordered=True.
    """
    if __shmdebug__: 
      print 'shm debugging'
      return self.map_debug(work, sequence, chunksize, ordered, star)
    L = len(sequence)
    if not hasattr(sequence, '__getitem__'):
      raise TypeError('can only take a slicable sequence')

    def worker(S, sequence, Q, i):
      self._local._rank = i
      dead = False
      error = None
      while True:
        begin, end = S.get()
        if begin is None: 
          S.task_done()
          break
        if dead: 
          Q.put((None, None))
          S.task_done()
          continue

        out = []
        try:
          for i in sequence[begin:end]:
            if star: out += [ work(*i) ]
            else: out += [ work(i) ]
        except Exception as e:
          error = (e, traceback.format_exc())
          Q.put(error)
          dead = True

        if not dead: Q.put((begin, out))
        S.task_done()
    P = []
    Q = self.QueueFactory()
    S = self.JoinableQueueFactory()

    i = 0

    N = 0
    while i < L:
      j = i + chunksize 
      if j > L: j = L
      S.put((i, j))
      i = j
      N = N + 1

    for i in range(self.np):
        S.put((None, i)) # sentinel

    # the slaves will not raise KeyboardInterrupt Exceptions
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    for i in range(self.np):
        p = self.SlaveFactory(target=worker, args=(S, sequence, Q, i))
        P.append(p)

    for p in P:
        p.start()

    signal.signal(signal.SIGINT, old)

    S.join()


#   the result is not sorted yet
    R = []
    error = []
    while N > 0:
      ind, r = Q.get()
      if isinstance(ind, Exception): 
        error += [(ind, r)]
      elif ind is None:
        # the worker dead, nothing done
        pass
      else:
        R.append((ind, r))
      N = N - 1

    # must clear Q before joining the Slaves or we deadlock.
    while not Q.empty():
      warn("unexpected extra queue item: %s" % str(Q.get()))
      
    i = 0
    alive = 1
    while alive > 0 and i < __timeout__:
      alive = 0
      for rank, p in enumerate(P):
        if p.is_alive():
          p.join(10)
          if p.is_alive():
            # we are in a serious Bug of sharedmem if reached here.
            warn("still waiting for slave %d" % rank)
            alive = alive + 1
      i = i + 1

    if alive > 0:
      warn("%d slaves alive after queue joined" % alive)
      
    # now report any errors
    if error:
      raise Exception('%d errors received\n' % len(error) + error[0][1])

    if ordered:
      heapq.heapify(R)
      chain = itertools.chain.from_iterable((heapq.heappop(R)[1] for i in range(len(R))))
    else:
      chain = itertools.chain.from_iterable((r[1] for r in R ))
    return numpy.array(list(chain))

  def map_debug(self, work, sequence, chunksize=1, ordered=False, star=False):
    if star: return [work(*x) for x in sequence]
    else: return [work(x) for x in sequence]

# Pickling is needed only for mp.Pool. Our pool is directly based on Process
# thus no need to pickle anything

def __unpickle__(ai, dtype):
  dtype = numpy.dtype(dtype)
  tp = ctypeslib._typecodes['|u1']
  # if there are strides, use strides, otherwise the stride is the itemsize of dtype
  if ai['strides']:
    tp *= ai['strides'][-1]
  else:
    tp *= dtype.itemsize
  for i in numpy.asarray(ai['shape'])[::-1]:
    tp *= i
  # grab a flat char array at the sharemem address, with length at least contain ai required
  ra = tp.from_address(ai['data'][0])
  buffer = ctypeslib.as_array(ra).ravel()
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
  def __init__(self):
    pass
  def __reduce__(self):
    return __unpickle__, (self.__array_interface__, self.dtype)

copy_reg.pickle(SharedMemArray, __pickle__, __unpickle__)

def empty_like(array, dtype=None):
  if dtype is None: dtype = array.dtype
  return empty(array.shape, dtype)

def empty(shape, dtype='f8'):
  """ allocates an empty array on the shared memory """
  dtype = numpy.dtype(dtype)
  tp = ctypeslib._typecodes['|u1'] * dtype.itemsize
  ra = RawArray(tp, int(numpy.asarray(shape).prod()))
  shm = ctypeslib.as_array(ra)
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

def copy(a):
  """ copies an array to the shared memory, use
     a = copy(a) to immediately dereference the old 'a' on private memory
   """
  shared = empty(a.shape, dtype=a.dtype)
  shared[:] = a[:]
  return shared

def wrap(a):
  return copy(a)

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

def __round_to_power_of_two(i):
  if i == 0: return i
  if (i & (i - 1)) == 0: return i
  i = i - 1
  i |= (i >> 1)
  i |= (i >> 2)
  i |= (i >> 4)
  i |= (i >> 8)
  i |= (i >> 16)
  i |= (i >> 32)
  return i + 1

def argsort(data, order=None):
  """
     parallel argsort, like numpy.argsort

     first call numpy.argsort on nchunks of data,
     then merge the returned arg.
     it uses 2 * len(data) * int64.itemsize of memory during calculation,
     that is len(data) * int64.itemsize in addition to the size of the returned array.
     the default chunksize (65536*16) gives a sorting time of 0.4 seconds on a single core 2G Hz computer.
     which is justified by the cost of spawning threads and etc.

     it uses an extra len(data)  * sizeof('i8') for the merging.
     we use threads because it turns out with threads the speed is faster(by 10%~20%)
     for sorting a 100,000,000 'f8' array, on a 16 core machine.
     
     TODO: shall try to use the inplace merge mentioned in 
            http://keithschwarz.com/interesting/code/?dir=inplace-merge.
  """

  from _mergesort import merge
  from _mergesort import reorderdtype
#  if len(data) < 64*65536: return data.argsort()

  if order: 
    newdtype = reorderdtype(data.dtype, order)
    data = data.view(newdtype)

  if cpu_count() <= 1: return data.argsort()

  nchunks = __round_to_power_of_two(cpu_count()) * 4

  arg1 = numpy.empty(len(data), dtype='i8')

  data_split = numpy.array_split(data, nchunks)
  sublengths = numpy.array([len(x) for x in data_split], dtype='i8')
  suboffsets = numpy.zeros(shape = sublengths.shape, dtype='i8')
  suboffsets[1:] = sublengths.cumsum()[:-1]

  arg_split = numpy.array_split(arg1, nchunks)

  with Pool(use_threads=True) as pool:
    def work(data, arg):
      arg[:] = data.argsort()
    pool.starmap(work, zip(data_split, arg_split))
  
  arg2 = numpy.empty(len(data), dtype='i8')

  def work(off1, len1, off2, len2, arg1, arg2, data):
    merge(data[off1:off1+len1+len2], arg1[off1:off1+len1], arg1[off2:off2+len2], arg2[off1:off1+len1+len2])

  while len(sublengths) > 1:
    with Pool(use_threads=True) as pool:
      pool.starmap(work, zip(suboffsets[::2], sublengths[::2], suboffsets[1::2], sublengths[1::2], repeat(arg1), repeat(arg2), repeat(data)))
    arg1, arg2 = arg2, arg1
    suboffsets = [x for x in suboffsets[::2]]
    sublengths = [x+y for x,y in zip(sublengths[::2], sublengths[1::2])]

  return arg1

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
    def work(arg, out):
      #needs numpy ticket #2156
      if len(out) == 0: return
      source.take(arg, axis=axis, out=out, mode=mode)
    pool.starmap(work, pool.zipsplit((indices, out)))
  return out
