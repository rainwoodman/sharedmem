"""
Dispatch your trivially parallizable jobs with sharedmem.

There are also sharedmem.argsort() and sharedmem.fromfile(),
which are the parallel equivalent of numpy.argsort() and numpy.fromfile().

Environment variable OMP_NUM_THREADS is used to determine the
default number of Slaves.

Two major components:

sharedmem.MapReduce and sharedmem.Parallel.

sharedmem.MapReduce is a hybrid of Map-Reduce and OpenMP parallel For, so
there are the goodies from both functional and procedural protocals.

sharedmem.Parallel is an incomplete implemenation of OpenMP with
multi-processaing (not with the multiprocessing module). This is experimental
and not well tested. parallel, forloop, barrier, master, critical, ordered are
implemented.

sharedmem.MapReduce has the following features:

1 Thread local storage regardless of backend(Thread or Process),
   just have to save attributes to 

   def work(jobid):
      pool.tls.myownvalue = myownvalue

  Note that it is a good idear to in general avoid using the Thread backend.

2 Critical section

   def work(jobid):
       do stuff in parallel
       with pool.critical:
          do stuff in critical section
       do more stuff in parallel

3 Ordered Execution 

   def work(jobid):
       do stuff that can be parallel
       with pool.ordered:
          do stuff that has to be ordered and serial
       do more stuff that can be parallel

4 Reduce operation.
  Reduce operation is executed on the Master process.
  reduce() is called after the return value from Slave is passed
  into Master. The return value of reduce is used to construct the
  final returned list.

   def work(jobid):
      return stuff

   def reduce(stuff):
      reduce stuff on the Master,
      this is serial

   pool.map(work, listofjobs, reduce) 

5 numpy like memory allocation of sharemed memory segments.

  x = sharedmem.empty(1000, dtype='f8')
  SharedMem segments are useful for communicating large chunk of 
  data from Slaves to Master.


Debugging:
  It is difficult to debug parallel code. There is a debugging mode
  where everything is run from the Master, and can be debugged.

    sharedmem.set_debug(True)

Backends:
  sharedmem has 2 parallel backends: Process and Threads.

  1. Processes.
    * Python code is executed in parallel. No GIL hassle.
    * Modification to varibales, including contents of numpy arrrays
      are copy on write. They do not show up in the Master.
    * Use sharedmem.empty() for arrays that needs to be synced.
  2. Threads
    * Python code is executed in serial. numpy/scipy functions
      are not fully GIL aware. Scipy uses some libraries that
      are very thread unfriendly. 
    * Modification to contents of variables shows up in Master.
    * Need to be very careful. Avoid using it in general.

Other Tools:
  * Sorting:
      A Parallel merge-sort with argsort. It uses a lot of memory.
      Use when appropriate.
"""

from core import *
from lib import *
