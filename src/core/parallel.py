import numpy
from multiprocessing.sharedctypes import RawValue
import threading
import signal
import Queue as queue
import heapq
import time

import backends


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
