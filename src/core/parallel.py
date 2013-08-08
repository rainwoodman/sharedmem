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

__all__ = ['MapReduce']

class MapReduce(object):
    def __init__(self, backend=backends.ProcessBackend, np=None):
        """ if np is 0, run in serial """
        self.backend = backend
        self._tls = self.backend.StorageFactory()
        if np is None:
            self.np = backends.cpu_count()
        else:
            self.np = np
        self.critical = self.backend.LockFactory()
        self.ordered = Ordered(self.backend, self._tls)

    def map(self, func, sequence, reduce=None, star=False):
        def realreduce(r):
            if reduce:
                if isinstance(r, tuple):
                    return reduce(*r)
                else:
                    return reduce(r)
            return r

        def realfunc(i):
            if star: return func(*i)
            else: return func(i)
        if self.np == 0:
            #Do this in serial
            return [realreduce(realfunc(i)) for i in sequence]

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
                if len(capsule) == 1:
                    i, = capsule
                    work = sequence[i]
                else:
                    i, work = capsule
                self.ordered.move(i)
                r = realfunc(work)
                pg.put(R, (i, r), master=False)

        pg = ProcessGroup(main=main, np=self.np, backend=self.backend)
        pg.start()

        L = []
        N = []
        def fetcher():
            count = 0
            while pg.is_alive():
                try:
                    capsule = R.get(timeout=1)
                except queue.Empty:
                    continue
                capsule = capsule[0], realreduce(capsule[1])
                heapq.heappush(L, capsule)
                count = count + 1
                if len(N) > 0 and count == N[0]: 
                    # if finished feeding see if all
                    # results have been obtained
                    return
        
        fetcher = threading.Thread(None, fetcher)
        fetcher.start()
        
        j = 0
        for i, work in enumerate(sequence):
            if not hasattr(sequence, '__getitem__'):
                pg.put(Q, (i, work))
            else:
                pg.put(Q, (i, ))
            j = j + 1
        N.append(j)

        for i in range(self.np):
            pg.put(Q, None)
        pg.join()
        fetcher.join()

        rt = []
        if len(L) > 0:
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
