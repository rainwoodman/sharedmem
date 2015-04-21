import os
import multiprocessing
import threading
import Queue as queue
from collections import deque
import traceback
import time
import gc
#logger = multiprocessing.log_to_stderr()
#logger.setLevel(multiprocessing.SUBDEBUG)

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

      On PBS/torque systems if OMP_NUM_THREADS is empty, we try to
      use the value of PBS_NUM_PPN variable.

      on some machines the physical number of cores does not equal
      the number of cpus shall be used. PSC Blacklight for example.

      Pool defaults to use cpu_count() slaves. however it can be overridden
      in Pool.
  """
  num = os.getenv("OMP_NUM_THREADS")
  if num is None:
      num = os.getenv("PBS_NUM_PPN")
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
        self.errorguard = threading.Thread(target=self._errorGuard)
        # this has to be from backend because the slaves will check
        # this variable.

        self.guardDead = backend.EventFactory()
        # each dead child releases one sempahore
        # when all dead guard will proceed to set guarddead
        self.semaphore = threading.Semaphore(0)
        self.JoinedProcesses = multiprocessing.RawValue('l')
        self.P = [
            backend.SlaveFactory(target=self._slaveMain,
                args=(rank,)) \
                for rank in range(np)
            ]
        self.G = [
            threading.Thread(target=self._slaveGuard,
                args=(rank, self.P[rank])) \
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
                print e
                self.Errors.put((e, traceback.format_exc()), timeout=0)
            except queue.Full:
                pass
        finally:
#            self.Errors.close()
#            self.Errors.join_thread()
            # making all slaves exit one after another
            # on some Linuxes if many slaves (56+) access
            # mmap randomly the termination of the slaves
            # run into a deadlock.
            while self.JoinedProcesses.value < rank:
                continue
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

    def _errorGuard(self):
        # this guard will kill every child if
        # an error is observed. We watch for this every 0.5 seconds
        # (errors do not happen very often)
        # if guardDead is set or killall is emitted, this will end immediately.
        while not self.guardDead.is_set():
            if not self.Errors.empty():
                self.killall()
                break
            # for python 2.6.x wait returns None XXX
            self.guardDead.wait(timeout=0.5)

    def _slaveGuard(self, rank, process):
        process.join()
        if isinstance(process, threading.Thread):
            pass
        else:
            if process.exitcode < 0 and process.exitcode != -5:
                e = Exception("slave process %d killed by signal %d" % (rank, -
                    process.exitcode))
                try:
                    self.Errors.put((e, ""), timeout=0)
                except queue.Full:
                    pass
        self.semaphore.release() 

    def _guardMain(self):
        # this guard will wait till all children are dead.
        # we then set the guardDead event
        def waitone(x):
            self.semaphore.acquire()
            self.JoinedProcesses.value = self.JoinedProcesses.value + 1
        map(waitone, self.G)
        self.guardDead.set()

    def start(self):
        self.JoinedProcesses.value = 0
        self.guardDead.clear()

        # collect the garbages before forking so that the left-over
        # junk won't throw out assertion errors due to
        # wrong pid in multiprocess.heap
        gc.collect()

        map(lambda x: x.start(), self.P)

        # p is alive from the moment start returns.
        # thus we can join them immediately after start returns.
        # guardMain will check if the slave has been
        # killed by the os, and simulate an error if so.
        map(lambda x: x.start(), self.G)
        self.errorguard.start()
        self.guard.start()

    def get_exception(self):
        return SlaveException(*self.Errors.get(timeout=0))

    def get(self, Q):
        """ Protected get. Get an item from Q.
            Will block. but if the process group has errors,
            raise an StopProcessGroup exception.

            A slave process will terminate upon StopProcessGroup.
            The master process shall read the error
        """
        while self.Errors.empty():
            try:
                return Q.get(timeout=1)
            except queue.Empty:
                if not self.is_alive():
                    raise StopProcessGroup
                else:
                    continue
        else:
            raise StopProcessGroup

    def put(self, Q, item):
        while self.Errors.empty():
            try:
                Q.put(item, timeout=1)
                return
            except queue.Full:
                if not self.is_alive():
                    raise StopProcessGroup
                else:
                    continue
        else:
            raise StopProcessGroup

    def is_alive(self):
        return not self.guardDead.is_set()

    def join(self):
        self.guardDead.wait()
        map(lambda x: x.join(), self.G)
        self.errorguard.join()
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

