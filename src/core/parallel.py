import numpy
from multiprocessing.sharedctypes import RawValue
import threading
import signal
import Queue as queue
import traceback
import heapq
import time
from collections import deque

import backends

__all__ = ['ProcessGroup']

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
        except Exception as e:
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

class MapReduce(object):
    def __init__(self, backend=backends.ProcessBackend, np=None):
        self.backend = backend
        self._tls = self.backend.StorageFactory()
        if np is None:
            self.np = backends.cpu_count()
        else:
            self.np = np
        self.critical = self.backend.LockFactory()
        self.ordered = Ordered(self.backend, self._tls)

    def map(self, func, sequence, reduce=None, star=False):
        Q = self.backend.QueueFactory(1)
        R = self.backend.QueueFactory(1)

        self.ordered.reset()
        def main(rank, pg):
            self._tls.rank = rank
            # get and put will raise SlaveException
            # and terminate the process.
            # the exception is muted in ProcessGroup,
            # as it will only be dispatched from master.
            while True:
                capsule = pg.get(Q, master=False)
                if capsule is None:
                    return
                i, work = capsule
                self.ordered.move(i)
                if star: r = func(*work)
                else : r = func(work)
                pg.put(R, (i, r), master=False)

        pg = ProcessGroup(main=main, np=self.np, backend=self.backend)
        pg.start()
        L = []

        N = []
        def fetcher():
            count = 0
            while pg.is_alive():
                try:
                    r = R.get(timeout=1)
                except queue.Empty:
                    continue
                heapq.heappush(L, r)
                count = count + 1
                if len(N) > 0 and count == N[0]: 
                    return

        
        fetcher = threading.Thread(None, fetcher)
        fetcher.start()
        
        j = 0
        for i, work in enumerate(sequence):
            pg.put(Q, (i, work))
            j = j + 1
        N.append(j)

        for i in range(self.np):
            pg.put(Q, None)
        pg.join()
        fetcher.join()

        rt = []
        for len(L) > 0:
            rt.append(heapq.heappop(L)[1])
        return rt

class Parallel(object):
    def __init__(self, backend=backends.ProcessBackend, np=None):
        self.backend = backend
        self._tls = self.backend.StorageFactory()
        if np is None:
            self.np = backends.cpu_count()
        else:
            self.np = np
        self.critical = self.backend.LockFactory()
        self.WorkQueue = self.backend.QueueFactory(1)
        self.ordered = Ordered(self.backend, self._tls)
    @property
    def debug(self):
        return backends.get_debug()

    @property
    def rank(self):
        return self._tls.rank

    def __call__(self, body):
        Q = self.backend.QueueFactory(self.np)
        def main(rank, pg):
            self._tls.rank = rank
            Q.put(body(self))

        pg = ProcessGroup(self.backend, main, self.np)
        pg.start()
        result = [pg.get(Q) for rank in range(self.np)]
        pg.join()
        return self

    def _ForStatic(self, range):
        rank = self.rank
        start = rank * len(range)// self.np
        end = (rank + 1) * len(range) // self.np
        for i in range[start:end]:
            yield i

    def _ForDynamic(self, range):
        if self.rank == 0:
            self.WorkQueue.put(0)
        while True:
            i = self.WorkQueue.get()
            i = i + 1
            if i <= len(range):
                self.WorkQueue.put(i)
                yield i - 1, range[i - 1]
                continue
            if i <= len(range) + self.np - 1:
                self.WorkQueue.put(i)
            break

    def For(self, range, schedule='static'):
        if schedule == 'static':
            _For = self._ForStatic
        if schedule == 'dynamic':
            _For = self._ForDynamic

        if self.rank == 0:
            self.ordered.reset()

        for i, work in _For(range):
            self.ordered.move(i)
            yield work
        
class Ordered(object):
    def __init__(self, backend, tls):
        self.event = backend.EventFactory()
        self.counter = RawValue('l')
        self.tls = tls

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
        self.event.set()
        self.counter.value = self.counter.value + 1

def main():
    import time
    import os
    import signal
    import numpy

    @Parallel(np=8, backend=backends.ThreadBackend)
    def body(par):
        for i in par.For(range(24), schedule='dynamic'):
            time.sleep(1)
            with par.ordered:
                print par.rank, i, i * i
            with par.critical:
                print par.rank, i, i * i
            if par.rank < 7:
                asfqewf
def main2():
    import time
    import os
    import signal
    import numpy
    m = MapReduce(np=8)
    def work(i):
    #    with m.ordered:
    #        time.sleep(1)
        print i
        return i
    print m.map(work, range(18))
main2()
