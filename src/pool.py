import numpy
from multiprocessing.sharedctypes import RawValue
import multiprocessing as mp
import threading
import signal
import Queue as queue
from warnings import warn
import traceback
import heapq

from tools import get_debug, cpu_count

class Pool:
  """
    with Pool() as p
      def work(a, b, c):
        pass
      p.starmap(work, zip(A, B, C))

    To use a Thread pool, pass use_threads=True
    there is a Lock accessible as 'with p.lock'

    Refer to the module document.
  """
  def __enter__(self):
    return self
  def __exit__(self, type, value, traceback):
    pass

  @property
  def iter(self):
    return self._tls.iter
  @property
  def rank(self):
    return self._tls.rank

  @property 
  def local(self):
    return self._tls.local

  def _init_threading(self):
      self.QueueFactory = queue.Queue
      self.JoinableQueueFactory = queue.Queue
      self.EventFactory = threading.Event
      def func(*args, **kwargs):
        slave = threading.Thread(*args, **kwargs)
        slave.daemon = True
        return slave
      self.SlaveFactory = func
      self.LockFactory = threading.Lock
      self._tls = threading.local()

  def _init_mp(self):
      self.QueueFactory = mp.Queue
      self.JoinableQueueFactory = mp.JoinableQueue
      self.EventFactory = mp.Event
      self.LockFactory = mp.Lock
      def func(*args, **kwargs):
        slave = mp.Process(*args, **kwargs)
        slave.daemon = True
        return slave
      self.SlaveFactory = func
      self._tls = lambda: None
     # master threads's rank is None
      self._tls.rank = None

  def __init__(self, np=None, use_threads=False):
    if np is None: np = cpu_count()
    self.np = np
    self.serial = False

    if use_threads:
        if threading.currentThread().name != 'MainThread':
            self.serial = True
            warn('nested TPool is avoided', stacklevel=2)
        self._init_threading()
    else:
        self._init_mp()

    self._tls.local = {}
    self.critical= self.LockFactory()
    self.stop = self.EventFactory()
    self.stop.clear()
    self.ordered = ordered(self)

  def starmap(self, work, sequence, reduce=None):
    return self.map(work, sequence, reduce=reduce, star=True)

  def map(self, workfunc, sequence, reduce=None, star=False):
    """
      calls workfunc on every item in sequence. 
    """
    if get_debug() or self.serial: 
      return self.map_debug(workfunc, sequence, reduce, star)

    if hasattr(sequence, "__getitem__"):
        indirect = True
    else:
        indirect = False

    def slave(S, Q, rank):
      self._tls.rank = rank
      while True:
        if self.stop.is_set():
           return
        try:
           workcapsule = S.get(timeout=1)
        except queue.Empty:
           continue
        if workcapsule is None: 
          Q.put((None, None))
          S.task_done()
          #print 'worker', rank, 'exit'
          break

        i, work = workcapsule
        self._tls.iter = i
       # print 'worker', rank, 'doing', i
        try:
          if indirect:
            work = sequence[i]
          if star: out = workfunc(*work)
          else: out = workfunc(work)
          Q.put((i, out))
          S.task_done()
        except Exception as e:
          Q.put((e, traceback.format_exc()))
          S.task_done()
        #print 'worker', rank, 'done', i

    P = []
    Q = self.QueueFactory(self.np)
    S = self.JoinableQueueFactory(1)#self.np)

#   the result is not sorted yet
    R = []
    error = []

    def feeder(S):
        for i, work in enumerate(sequence):
          # we do not send work through pipe
          # because work may be large array and pickling them is
          # painful
          while True:
            try:
              if len(error) > 0: 
                  # if error detected
                  # just stops feeding at all.
                  # we will die with an exception
                  self.stop.set()
                  return
              if indirect:
                  S.put((i, None), timeout=1)
              else:
                  S.put((i, work), timeout=1)
              break
            except queue.Full:
              continue
        for rank in range(self.np):
            S.put(None) # sentinel

    def fetcher(Q):
        L = self.np
        while L > 0:
          # see if the total number of alive processes
          # match the number of processes that hasn't
          # finished gracefully
          # if unmatch and 
          # the queue is still empty,(no worker finished 
          # in the meanwhile)
          # some workers have been killed by OS
          if numpy.sum([p.is_alive() for p in P]) < L \
               and Q.empty():
            ind = Exception("Some processes killed unexpectedly\n")
            r = "Unknown Traceback"
          else:
            try:
              ind, r = Q.get(timeout=2)
            except queue.Empty:
              continue
          if ind is None:
            # the worker dead, nothing done
            #print 'gracefully worker finished', L
            L = L - 1
          elif isinstance(ind, Exception): 
            #print 'worker errored', L
            L = L - 1
            error.append((ind, r))
            # after first error is received, report and stop
            # monitoring the queue
            break
          else:
            # success
            try:
              if reduce is not None:
                if isinstance(r, tuple):
                  r = reduce(*r)
                else:
                  r = reduce(r)
              R.append((ind, r))
            except Exception as e:
              error.append((e, traceback.format_exc()))
              break
        #print 'fetcher ended', L


    # the slaves will not raise KeyboardInterrupt Exceptions
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)

    for rank in range(self.np):
        p = self.SlaveFactory(target=slave, args=(S, Q, rank))
        P.append(p)

    for p in P:
        p.start()

    fetcher = threading.Thread(target=fetcher, args=(Q,))
    fetcher.start()
    feeder = threading.Thread(target=feeder, args=(S,))
    feeder.start()

    signal.signal(signal.SIGINT, old)

    while feeder.is_alive():
        try:
            feeder.join(2)
        except (KeyboardInterrupt, SystemExit) as e:
            error.append((e, traceback.format_exc()))
            #print error

    #print 'feeder joined'
    while fetcher.is_alive():
    #    print 'fetcher joining'
        fetcher.join(timeout=2)

    # now report any errors
    if len(error) > 0:
        raise Exception(str(error[0][0]) + "\ntraceback:\n" + error[0][1])
    else:
      # must clear Q before joining the Slaves or we deadlock.
      while not Q.empty():
        raise Exception("unexpected extra queue item: %s" % str(Q.get()))

      for rank, p in enumerate(P):
        p.join()

    heapq.heapify(R)
    return [ heapq.heappop(R)[1] for i in range(len(R))]

  def map_debug(self, work, sequence, reduce=None, star=False):
    def realreduce(x):
      if reduce is None: return x
      if isinstance(x, tuple):
        return reduce(*x)
      else:
        return reduce(x)
    if star: return [realreduce(work(*x)) for x in sequence]
    else: return [realreduce(work(x)) for x in sequence]

class TPool(Pool):
    def __init__(self, np=None):
        Pool.__init__(self, use_threads=True, np=np)


class ordered(object):
    def __init__(self, pool):
        self.pool = pool
        self.event = pool.EventFactory()
        self.count = RawValue('l')
        self.count.value = 0
        self.event.set()
    def __enter__(self):
        while True:
            self.event.wait()
            if self.count.value == self.pool.iter:
                self.event.clear()
                return self
    def __exit__(self, type, value, traceback):
        self.count.value = self.count.value + 1
        self.event.set()

