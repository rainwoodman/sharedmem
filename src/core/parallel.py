import numpy
from multiprocessing.sharedctypes import RawValue
import traceback as tb
import signal
import time
import os 
import backends
from multiprocessing import Lock, Event
from multiprocessing import Semaphore
from multiprocessing.queues import SimpleQueue
from threading import Thread
from memory import empty
class ParallelException(Exception):
    pass

class ErrorMonitor(object):
    def __init__(self):
        self.pipe = SimpleQueue()
        self.message = None
    def main(self):
        while True:
            message = self.pipe.get()
            if message[0] == 'Q':
                break
            else:
                self.message = message[1:]
    
    def haserror(self):
        """ master only """
        return self.message is not None
    def start(self):
        """ master only """
        self.thread = Thread(target=self.main)
        self.thread.start()
    def join(self):
        """ master only """
        self.pipe.put('Q')
        self.thread.join()
        self.thread = None
        if self.message is not None:
            raise ParallelException(self.message)

    def slaveraise(self, error, traceback):
        """ slave only """
        self.pipe.put('E' + str(error) + tb.format_exc())


class Parallel(object):
    def __init__(self, *args, **kwargs):
        self.np = kwargs.get('np', backends.cpu_count())
        self.rank = 0
        self.var = Var()
        self._variables = args
    def _ismaster(self):
        return self.rank == 0

    def _fork(self):
        self._children = []
        self.rank = 0
        for i in range(self.np - 1):
            if not self._ismaster(): continue
            pid = os.fork()
            if pid != 0:
                self._children.append(pid)
            else:
                self.rank = i + 1

    def __enter__(self):
        self.critical = Lock()
        self._alive = True
        self._errormessage = None
        self._errormon = ErrorMonitor() 
        self._shared = empty((),
                dtype=[
                    ('ordered', 'intp'),
                    ('barrier', 'intp')])
        self._event = Event()
        self._shared['barrier'] = 0
        self._barrier = Barrier(self.np, self._shared['barrier'])
        class Ordered(BaseOrdered):
            turnstile = Semaphore(1)
            done = self._shared['ordered'][...]
        self._Ordered = Ordered
        
        for param in self._variables:
            param.beforefork(self)
        self._fork()
        for param in self._variables:
            param.afterfork(self)
        if self._ismaster():
            self._errormon.start()
        return self

    def __exit__(self, type, exception, traceback):
        if not self._ismaster(): 
            # put error to the pipe
            if type is not None:
                # need to make sure each (per) message 
                # won't block the pipe!
                self._errormon.slaveraise(exception, traceback)
            os._exit(0)
        else:
            if type is not None:
                for pid in self._children:
                    os.kill(pid, 6)
            n = 0
            while n < self.np - 1:
                pid, status = os.wait()
                n = n + 1
            self._errormon.join()
            for param in self._variables:
                if isinstance(param, Reduction):
                    param.reduce(self)
        self.critical = None

    def barrier(self):
        self._barrier.wait()

    def forloop(self, range, ordered=False):
        return ForLoop(range, ordered, self)

class Barrier:
    def __init__(self, n, count):
        self.n = n
        self.count = count
        self.count[...] = 0
        self.mutex = Semaphore(1)
        self.turnstile = Semaphore(0)
        self.turnstile2 = Semaphore(0)
    def phase1(self):
        self.mutex.acquire()
        self.count[...] += 1
        if self.count == self.n:
            [self.turnstile.release() for i in range(self.n)]
        self.mutex.release()
        self.turnstile.acquire()
    def phase2(self):
        self.mutex.acquire()
        self.count[...] -= 1
        if self.count == 0:
            [self.turnstile2.release() for i in range(self.n)]
        self.mutex.release()
        self.turnstile2.acquire()
    def wait(self):
        if self.n == 0: return
        self.phase1()
        self.phase2()

class ForLoop(object):
    def __init__(self, range, ordered, parallel):
        start = parallel.rank * len(range) // parallel.np
        end = (parallel.rank + 1)* len(range) // parallel.np
        self.range = range[start:end]
        self.iter = numpy.empty((), dtype='intp')
        self.iter[...] = 0
        if parallel._ismaster():
            self._haserror = parallel._errormon.haserror
        else:
            self._haserror = lambda : False
        if ordered:
            self.ordered = parallel._Ordered(self.iter)
    def __iter__(self):
        for i in self.range:
            if self._haserror():
                break
            self.iter[...] = i
            yield i
        self._haserror = None

class BaseOrdered(object):
    """subclass and set kls.done, cls.event to sharedmem objects """
    def __init__(self, iterref):
        self.done[...] = 0
        self.iterref = iterref

    def __enter__(self):
        while self.iterref != self.done:
            pass

        self.turnstile.acquire()
        return self
    def __exit__(self, *args):
        self.done[...] += 1
        self.turnstile.release()

class Var(object):
    def __init__(self):
        self.__dict__['map'] = {}
    def _register(self, varset, index):
        self.__dict__['map'][index] = varset
    def __getattr__(self, index):
        set = self.__dict__['map'][index]
        return set[index]
    def __setattr__(self, index, value):
        set = self.__dict__['map'][index]
        set[index] = value

class VarSet(object):
    """ overridebefore fork and afterfork"""
    def __init__(self, **kwargs):
        self._input = [(key, numpy.asarray(kwargs[key]))
                for key in kwargs]
        self._dtype = [(key, (item.dtype, item.shape)) for key, item in self._input]

    def beforefork(self, parallel):
        """ allocate self.data with dtype """
        pass

    def afterfork(self, parallel):
        self._input = None
        self._dtype = None
        for key in self:
            parallel.var._register(self, key)

    def __iter__(self):
        return iter(self.data.dtype.names)
    def __getitem__(self, index):
        return self.data[index]
    def __setitem__(self, index, value):
        self.data[index][...] = value

class Shared(VarSet):
    def beforefork(self, parallel):
        self.data = empty((), self._dtype)
        for key, value in self._input:
            self.data[key] = value

class Private(VarSet):
    def beforefork(self, parallel):
        self.data = numpy.empty((), self._dtype)
        for key, value in self._input:
            self.data[key] = value

class Reduction(VarSet):
    def __init__(self, ufunc, **kwargs):
        self._ufunc = ufunc
        VarSet.__init__(self, **kwargs)

    def beforefork(self, parallel):
        self._fulldata = empty(parallel.np, self._dtype)

    def afterfork(self, parallel):
        self.data = self._fulldata[parallel.rank]
        for key, value in self._input:
            self.data[key] = value
        VarSet.afterfork(self, parallel)

    def reduce(self, parallel):
        if parallel._ismaster():
            for key in self:
                self[key] = self._ufunc.reduce(self._fulldata[key], axis=0)
        else:
            self.data = self._fulldata[0].view(type=Shared)

def testraiseordered():
    with Parallel(
            Reduction(numpy.add, a=[0, 0])
            ) as p:
        r = p.forloop(range(20), ordered=True)
        for i in r:
            with r.ordered:
                p.var.a += numpy.array([i, i * 10])
                if i == 19:
                    raise Exception('raised at i == 19')
    assert (p.var.a == [190, 1900]).all()

def testraisecritical():
    with Parallel(
            Reduction(numpy.add, a=[0, 0])
            ) as p:
        r = p.forloop(range(20), ordered=True)
        for i in r:
            with p.critical:
                p.var.a += numpy.array([i, i * 10])
                if i == 19:
                    raise Exception('raised at i == 19')
    assert (p.var.a == [190, 1900]).all()

def testreduction():
    with Parallel(
            Reduction(numpy.add, a=[0, 0])
            ) as p:
        r = p.forloop(range(20), ordered=True)
        for i in r:
            p.var.a += numpy.array([i, i * 10])
    assert (p.var.a == [190, 1900]).all()

def testprivate():
    truevalue = numpy.zeros(2)
    with Parallel(
            Private(a=[0, 0])
            ) as p:
        r = p.forloop(range(100), ordered=True)
        for i in r:
            p.var.a += numpy.array([i, i * 10])
            if p._ismaster(): 
                truevalue += numpy.array([i, i * 10])
    assert (p.var.a == truevalue).all()

def testshared():
    with Parallel(
            Shared(a=[0, 0]),
            Reduction(numpy.add, b=[0, 0])
            ) as p:
        r = p.forloop(range(100), ordered=True)
        for i in r:
            with p.critical:
                p.var.a += numpy.array([i, i * 10])
            p.var.b += numpy.array([i, i * 10]) 
        
    assert (p.var.a == p.var.b).all()

def testbarrier():
    now = time.time()
    with Parallel(
            Shared(a=[0, 0]),
            Reduction(numpy.add, b=[0, 0])
            ) as p:
        time.sleep(p.rank * 0.01)
        p.barrier()
        p.barrier()
        
    print time.time() - now 
def main():
    testreduction()
    testprivate()
    testshared()
    testbarrier()
    try:
        testraisecritical()
    except ParallelException as e:
        print e
        pass
    try:
        testraiseordered()
    except ParallelException as e:
        print e
        pass

if __name__ == '__main__':
    main()
