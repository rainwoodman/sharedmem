import os
import multiprocessing
import threading
import Queue as queue

__shmdebug__ = False
__all__ = ['set_debug', 'get_debug', 'total_memory', 'cpu_count', 'ThreadBackend',
        'ProcessBackend', 'SlaveException', 'ProcessGroupFinished',
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

class ProcessGroupFinished(Exception):
    def __init__(self):
        Exception.__init__(self, "ProcessGroupFinished")

class ProcessGroup(object):
    def __init__(self, backend, main, np):
        self.Errors = backend.QueueFactory(1)
        self.main = main
        self.guard = threading.Thread(target=self._guardMain)

        self.guardDead = backend.EventFactory()
        self.P = [
            backend.SlaveFactory(target=self._slaveMain,
                args=(rank,)) \
                for rank in range(np)
            ]
        return

    def _slaveMain(self, rank):
        try:
            self.main(rank, self)
        except backends.SlaveException as e:
            pass
        except backends.ProcessGroupFinished as e:
            pass
        except BaseException as e:
            try:
                self.Errors.put((e, traceback.format_exc()), timeout=0)
            except queue.Full:
                pass
        finally:
            pass

    def killall():
        for p in self.P:
            if not p.is_alive(): continue
            if isinstance(p, threading.Thread): p.join()
            else: p.terminate()

    def _guardMain(self):
        Nalive = numpy.sum([p.is_alive() for p in self.P])
        q = deque(self.P)
        while self.Errors.empty() \
          and len(q) > 0:
            p = q.popleft()
            p.join(timeout=1)
            if p.is_alive(): q.append(p)
            if isinstance(p, threading.Thread): continue
            unexpected = numpy.sum([p.exitcode < 0 \
                    for p in self.P if not p.is_alive()])
            if unexpected > 0:
                e = Exception("slave process killed by signal %d" % -p.exitcode)
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

    def get(self, Q, master=True):
        while self.Errors.empty():
            if not self.is_alive():
                raise backends.ProcessGroupFinished
            try:
                return Q.get(timeout=1)
            except queue.Empty:
                continue
        else:
            if master:
                raise backends.SlaveException(*self.Errors.get())
            else:
                raise backends.ProcessGroupFinished

    def put(self, Q, item, master=True):
        while self.Errors.empty():
            if not self.is_alive():
                raise backends.ProcessGroupFinished
            try:
                Q.put(item, timeout=1)
                return
            except queue.Full:
                continue
        else:
            if master:
                raise backends.SlaveException(*self.Errors.get())
            else:
                raise backends.ProcessGroupFinished

    def is_alive(self):
        return not self.guardDead.is_set()

    def join(self):
        while self.is_alive():
            self.guardDead.wait(timeout=0.1)
            if not self.Errors.empty():
                raise backends.SlaveException(*self.Errors.get())
        self.guard.join()


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

