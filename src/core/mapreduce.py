import backends
import threading
import heapq
import gc
import os
import Queue as queue
__all__ = ['MapReduce', 'MapReduceByThread']

def MapReduceByThread(np=None):
    return MapReduce(backend=backends.ThreadBackend, np=np)

class MapReduce(object):
    def __init__(self, backend=backends.ProcessBackend, np=None):
        """ if np is 0, run in serial """
        self.backend = backend
        if np is None:
            self.np = backends.cpu_count()
        else:
            self.np = np

    def main(self, pg, Q, R, sequence, realfunc):
        # get and put will raise SlaveException
        # and terminate the process.
        # the exception is muted in ProcessGroup,
        # as it will only be dispatched from master.
        while True:
            capsule = pg.get(Q, reraise=False)
            if capsule is None:
                return
            if len(capsule) == 1:
                i, = capsule
                work = sequence[i]
            else:
                i, work = capsule
            self.ordered.move(i)
            r = realfunc(work)
            pg.put(R, (i, r), reraise=False)


    def __enter__(self):
        self.critical = self.backend.LockFactory()
        self.ordered = backends.Ordered(self.backend)
        return self

    def __exit__(self, *args):
        self.ordered = None
        pass

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

        pg = backends.ProcessGroup(main=self.main, np=self.np,
                backend=self.backend,
                args=(Q, R, sequence, realfunc))

        pg.start()

        L = []
        N = []
        def feeder(pg, Q, N):
            #   will fail silently if any error occurs.
            j = 0
            try:
                for i, work in enumerate(sequence):
                    if not hasattr(sequence, '__getitem__'):
                        pg.put(Q, (i, work), reraise=False)
                    else:
                        pg.put(Q, (i, ), reraise=False)
                    j = j + 1
                N.append(j)

                for i in range(self.np):
                    pg.put(Q, None, reraise=False)
            except:
                return
        

        feeder = threading.Thread(None, feeder, args=(pg, Q, N))
        feeder.start() 

        # we run fetcher on main thread to catch exceptions
        # raised by reduce 
        count = 0
        try:
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
                    break
            rt = []
            while len(L) > 0:
                rt.append(heapq.heappop(L)[1])
            pg.join()
            feeder.join()
            return rt
        except Exception as e:
            pg.Errors.put([e, ''])
            raise 

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
