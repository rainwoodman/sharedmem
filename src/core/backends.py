import os
import multiprocessing
import threading
import Queue as queue

__shmdebug__ = False
__all__ = ['set_debug', 'get_debug', 'total_memory', 'cpu_count', 'ThreadBackend',
        'ProcessBackend', 'SlaveException', 'ProcessGroupFinished']

def set_debug(flag):
  """ in debug mode (flag==True), no slaves are spawn,
      rather all work are done in serial on the master thread/process.
      so that if the worker throws out exceptions, debugging from the main
      process context is possible. (in iptyhon, with debug magic command, eg)
  """
  global __shmdebug__
  __shmdebug__ = flag

def get_debug():
  global __shmdebug__
  return __shmdebug__

def total_memory():
  """ the amount of memory available for use.
      default is the Free Memory entry in /proc/meminfo """
  with file('/proc/meminfo', 'r') as f:
      for line in f:
          words = line.split()
          if words[0].upper() == 'MEMTOTAL:':
                return int(words[1]) * 1024
  raise IOError('MemTotal unknown')
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
    return multiprocessing.cpu_count()

class SlaveException(Exception):
    def __init__(self, e, tracebackstr):
        Exception.__init__(self, "%s\n%s" % (str(e), tracebackstr))

class ProcessGroupFinished(Exception):
    def __init__(self):
        Exception.__init__(self, "ProcessGroupFinished")

class ThreadBackend:
      QueueFactory = staticmethod(queue.Queue)
      JoinableQueueFactory = staticmethod(queue.Queue)
      EventFactory = staticmethod(threading.Event)
      @staticmethod
      def SlaveFactory(*args, **kwargs):
        slave = threading.Thread(*args, **kwargs)
        slave.daemon = True
        return slave
      LockFactory = staticmethod(threading.Lock)
      StorageFactory = staticmethod(threading.local)

class ProcessBackend:
      QueueFactory = staticmethod(multiprocessing.Queue)
      JoinableQueueFactory = staticmethod(multiprocessing.JoinableQueue)
      EventFactory = staticmethod(multiprocessing.Event)
      LockFactory = staticmethod(multiprocessing.Lock)
      @staticmethod
      def SlaveFactory(*args, **kwargs):
        slave = multiprocessing.Process(*args, **kwargs)
        slave.daemon = True
        return slave
      @staticmethod
      def StorageFactory():
          return lambda:None

