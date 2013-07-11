"""
Dispatch your trivially parallizable jobs with sharedmem.

There are also sharedmem.argsort() and sharedmem.fromfile(),
which are the parallel equivalent of numpy.argsort() and numpy.fromfile().

Environment variable OMP_NUM_THREADS is used to determine the
default number of Slaves.

output = sharedmem.empty(8)
output2 = sharedmem.empty(8)

with sharedmem.Pool() as pool:
  def work(arg1):
    # do some work, return some value or not.
    output2[arg1] = pool.rank
    with pool.critical:
      output[...] += arg2
    with pool.ordered:
      do stuff in order
    return a, b, c
  def reduce(a, b, c):
    reduce result
    return d
  pool.map(work, ...., reduce)

output2 will be an array of the ranks,
and output will be 36, 36, ....

Pool:
  pool.rank, 
  pool.np, 
  pool.local,   local TLS storage
  pool.critical,   context manager for critical section
  pool.ordered,   context manager for ordered section

  pool.map(work, list, reduce) maps the parameters to work directly
  pool.starmap(work, list, reduce) maps the parameters as an argument list.

  work is run on the slaves in random order

  reduce is run on the master, in random order
  if list is an iterable, its content is pickled and passed to the 
  slaves.

  if list can be directly indexed, only the index of the item is 
  passed to the slaves. the slaves relies on inherited SHM to access
  the items in the list.

  The arguments of reduce is the same as the
  return value of work.

Debugging:
  sharedmem.set_debug(True)

  in debugging mode no Slaves are spawned. all work is done in the
  Master, thus the work function can be debugged.

Backends:
  sharedmem has 2 parallel backends: Process and Threads.

  If you do not write to array objects and all lengthy jobs
  releases GIL, then the two backends are equivalent. 

  1. Processes. 
  * with sharedmem.Pool() as pool:
  * There is no need ensure lengthy calculation needs to release
    the GIL.
  * Any numpy array that needs to be written by
    the slaves needs to reside on the shared memory segments. 
    numpy arrays on the shared memory has type SharedMemArray.

  * SharedMemArray are allocated with
      sharedmem.empty()
    or copied from local numpy array with
      sharedmem.copy()

    If possible use empty() for huge data sets.
    Because with copy() the data has to be copied and memory usage
    is (at least temporarily doubled)

  2. Threads
  * with sharedmem.TPool()
  * Slaves can write to ordinary numpy arrays.
  * need to ensure lengthy calculation releases the GIL.

"""

import multiprocessing as mp
import numpy

from pool import Pool, TPool
from tools import set_debug, cpu_count
from array import empty, empty_like, copy

from sort import argsort
from sort import searchsorted

