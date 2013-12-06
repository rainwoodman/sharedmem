import os
import multiprocessing
import threading
import Queue as queue
from collections import deque
import traceback

__shmdebug__ = False
__all__ = ['set_debug', 'get_debug', 'total_memory', 'cpu_count', 'ThreadBackend',
        'ProcessBackend', 'SlaveException', 'StopProcessGroup',
        'ProcessGroup']

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

class StopProcessGroup(Exception):
    """ StopProcessGroup will terminate the slave process/thread """
    def __init__(self):
        Exception.__init__(self, "StopProcessGroup")

class ProcessGroup(object):
    def __init__(self, backend, main, np, args=()):
        self.Errors = backend.QueueFactory(1)
        self._tls = backend.StorageFactory()
        self.main = main
        self.args = args
        self.guard = threading.Thread(target=self._guardMain)

        self.guardDead = backend.EventFactory()
        self.P = [
            backend.SlaveFactory(target=self._slaveMain,
                args=(rank,)) \
                for rank in range(np)
            ]
        return

    def _slaveMain(self, rank):
        self._tls.rank = rank
        try:
            self.main(self, *self.args)
        except SlaveException as e:
            raise RuntimError("slave exception shall never be caught by a slave")
        except StopProcessGroup as e:
            pass
        except BaseException as e:
            try:
                self.Errors.put((e, traceback.format_exc()), timeout=0)
            except queue.Full:
                pass
        finally:
            pass

    def killall(self):
        for p in self.P:
            if not p.is_alive(): continue
            try:
                if isinstance(p, threading.Thread): p.join()
                else: os.kill(p._popen.pid, 5)
            except Exception as e:
                print e
                continue

    def _guardMain(self):
        Nalive = sum([p.is_alive() for p in self.P])
        q = deque(self.P)
        while self.Errors.empty() \
          and len(q) > 0:
            p = q.popleft()
            p.join(timeout=1)
            if p.is_alive(): q.append(p)
            if isinstance(p, threading.Thread): continue
            unexpected = sum([p.exitcode < 0 and p.exitcode != -2 \
                    for p in self.P if not p.is_alive()])
            if unexpected > 0:
                e = Exception("slave process killed by signal %s" %
                        str([-p.exitcode for p in self.P if not p.is_alive()]))
                try:
                    self.Errors.put((e, ""), timeout=0)
                except queue.Full:
                    pass
                self.killall()
        self.guardDead.set()

    def start(self):
        self.guardDead.clear()

        map(lambda x: x.start(), self.P)

        # p is alive from the moment start returns.
        # thus we can join them immediately after start returns.
        # guardMain will check if the slave has been
        # killed by the os, and simulate an error if so.
        self.guard.start()

    def get(self, Q, reraise=True):
        """ get an item from Q,
            if an error is detected, 
                raise StopProcessGroup if reraise is False
                reraise the detected Error if reraise if True
        """
        while self.Errors.empty():
            if not self.is_alive():
                raise StopProcessGroup
            try:
                return Q.get(timeout=1)
            except queue.Empty:
                continue
        else:
            if reraise:
                raise SlaveException(*self.Errors.get())
            else:
                raise StopProcessGroup

    def put(self, Q, item, reraise=True):
        while self.Errors.empty():
            if not self.is_alive():
                raise StopProcessGroup
            try:
                Q.put(item, timeout=1)
                return
            except queue.Full:
                continue
        else:
            if reraise:
                raise SlaveException(*self.Errors.get())
            else:
                raise StopProcessGroup

    def is_alive(self):
        return not self.guardDead.is_set()

    def join(self):
        self.guardDead.wait()
        self.guard.join()
        if not self.Errors.empty():
            raise SlaveException(*self.Errors.get())

class Ordered(object):
    def __init__(self, backend):
      #  self.counter = lambda : None
        #multiprocessing.RawValue('l')
        self.event = backend.EventFactory()
        self.counter = multiprocessing.RawValue('l')
        self.tls = backend.StorageFactory()

    def reset(self):
        self.counter.value = 0
        self.event.set()

    def move(self, iter):
        self.tls.iter = iter

    def __enter__(self):
        while self.counter.value != self.tls.iter:
            self.event.wait() 
        self.event.clear()
        return self

    def __exit__(self, *args):
        # increase counter before releasing the value
        # so that the others waiting will see the new counter
        self.counter.value = self.counter.value + 1
        self.event.set()


class ThreadBackend:
      QueueFactory = staticmethod(queue.Queue)
      EventFactory = staticmethod(threading.Event)
      LockFactory = staticmethod(threading.Lock)
      StorageFactory = staticmethod(threading.local)
      @staticmethod
      def SlaveFactory(*args, **kwargs):
        slave = threading.Thread(*args, **kwargs)
        slave.daemon = True
        return slave

class ProcessBackend:
      QueueFactory = staticmethod(multiprocessing.Queue)
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

