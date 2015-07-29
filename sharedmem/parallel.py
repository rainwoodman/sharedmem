"""
   OpenMP like multiprocessing

   Example:

        with Parallel(
                Reduction(numpy.add, a=[0, 0])
                ) as p:
            for i, ordered in p.forloop(range(20), 
                    ordered=True, schedule='dynamic'):
                with ordered:
                    p.var.a += numpy.array([i, i * 10])
                    if i == 19:
                        raise ValueError('raised at i == 19')

   1 All variables are by default 'Private': this differ from OpenMP

   2 To access variables defined for the Parallel scope, use p.var

   3 This is too slow to be useful as I can imagine. 
     Looping in Python is horriblly slow as we know it
     and we use iterators to yield values, plus slow posix semaphores.

   4 forloop returns the iterator for the variable if ordered == False
     forloop returns the iterator variable and an Ordered object if
     ordered==True.

   5 DeadLocks: 
        Tested pretty extensively, but 
        there still may be some corner cases. 

        Biggest bunch I have fixed are because we rely on
        an asynchronized Exception(AE) to jump out of a parallel scope
        when there are exceptions from slave processes.
        In Python AE is unsafe, even with the new 'with' statement. 

        As a fix, we limit the AE only between __enter__ and __exit__,
        and the only nontrivial object used is Semaphore. In case of
        unprotected and failed Semaphore release, we release them one
        more time from another thread, guareenteed to be safe from 
        the AE we use (signal AE)

   

   Comparing similiar features between Parallel and OpenMP standard:

        Parallel               vs  openmp construct

        with Parallel() as p:           omp parallel 
            xxxx
            p.rank                      omp_get_thread_num()
            p.num_threads               omp_get_num_threads()

            p.var.a:                    a referes to the variable 'a' 
                                        decleared with Private, Shared or Reduction

            if p.master:                    omp master 
                xxxx
            for i in p.forloop():           omp for
                xxxxj 

            p.barrier()                       omp barrier

            r = p.forloop(ordered=True)     omp for
            for i in r:
                with r.ordered:              omp ordered
                    xxxx
            with p.critical:                 omp critical
                xxxx 
"""

__author__ = "Yu Feng"
__email__ = "rainwoodman@gmail.com"

import numpy
import traceback as tb
import signal
import time
import os 
import pickle
import signal

from multiprocessing import Lock
from multiprocessing.synchronize import Semaphore
from multiprocessing.queues import SimpleQueue
from threading import Thread

from . import sharedmem

__all__ = ['Parallel', 'ParallelException']

class ParallelException(Exception):
    """When a Slave process is unexpectedly killed (by the OS, eg, OOM)"""
    pass

class LongJump(BaseException):
    """Convert SIGTRAP to an python exception so that the
       parallel section will jump to __exit__.

       We use SIGTRAP to notify the master and slave that an exception
       has occured. 

       A slave process that received SIGTRAP will exit immediately.
       (We rely on the OS for that)

       LongJump is called by errormon upon detection of an error.
       (ran in errormon context)

       Python signal handler is executed at the 'next' python 
       executation point after the signal is received.

       To ensure master jumps to __exit__ we make sure the code runs
       to the next bytecode asap by 
       We forcefully release all Semaphores, after ensuring all slaves
       are killed.
    """
    muted = False
    parallel = None
    @staticmethod
    def handler(a, b):
        tb = []
        while b is not None:
            tb.append('%d %s %d' % (b.f_lineno, b.f_code.co_filename,
                os.getpid()))
            b = b.f_back
        raise LongJump('\n'.join(tb))
    @classmethod
    def listen(kls, parallel):
        kls.parallel = parallel
        signal.signal(signal.SIGTRAP, kls.handler)
        kls.muted = False
    @classmethod
    def mute(kls):
        if not kls.muted:
            kls.muted = True
            signal.signal(signal.SIGTRAP, signal.SIG_IGN)
    @staticmethod
    def longjump():
        LongJump.parallel._slavemon.kill_all()
        # first kill slaves, then master
        os.kill(os.getpid(), signal.SIGTRAP)

        # this line will always run becuase
        # longjump is called from non-main thread
        # signal is trapped on main-thread.
        # will will ensure all Semaphores are released
        # so that the master won't stuck
        # as all slaves are dead, we just need to
        # increase the semaphores enough to raise the master
        LongJump.parallel._cleanup()


class SlaveMonitor:
    def __init__(self, errormon):
        self.children = []
        self.errormon = errormon

    def main(self):
        n = 0
        N = len(self.children)
        while n < N:
            pid, status = os.wait()
            n = n + 1
            self.children.remove(pid)
            if status != 0 and status != signal.SIGTRAP:
                self.errormon.slaveraise(ParallelException, 
                        ParallelException("slave %d died unexpected: %d" % (pid,
                            status)), None)
    def notechild(self, pid):
        self.children.append(pid)
    def kill_all(self):
        """kill all slaves and reap the monitor """
        for pid in self.children:
            try:
                os.kill(pid, signal.SIGTRAP)
            except OSError:
                continue
        self.join()

    def start(self):
        self.thread = Thread(target=self.main)
        self.thread.daemon = True
        self.thread.start()

    def join(self):
        """ master only """
        try:
            self.thread.join()
        except:
            pass
        finally:
            self.thread = None

class ErrorMonitor:
    def __init__(self):
        self.pipe = SimpleQueue()
        self.message = None

    def main(self):
        while True:
            message = self.pipe.get()
            if message != 'Q':
                self.message = message[1:]
                LongJump.longjump()
                break
            else:
                self.pipe = None
                break
                    
    def haserror(self):
        """ master only """
        return self.message is not None
    def start(self):
        """ master only """
        self.thread = Thread(target=self.main)
        self.thread.daemon = True
        self.thread.start()
    def join(self):
        """ master only """
        try:
            self.pipe.put('Q')
            self.thread.join()
        except:
            pass
        finally:
            self.thread = None

    def slaveraise(self, type, error, traceback):
        """ slave only """
        message = 'E' * 1 + pickle.dumps((type,
            ''.join(tb.format_exception(type, error, traceback))))
        if self.pipe is not None:
            self.pipe.put(message)


class Parallel(object):
    """

        **kwargs: 
             num_threads: number of processes (default to OMP_NUM_THREADS)

        *args:
            Private(name=value, ....)
            Shared(name=value, ....)
            Reduction(ufunc, name=value, ....)

    """
    def __init__(self, *args, **kwargs):
        self.num_threads = kwargs.get('num_threads', sharedmem.cpu_count())
        self.var = Var()
        self.rank = 0
        self.master = True
        self._variables = args
        self._children = []


    def _fork(self):
        self.rank = 0
        self.master == True
        for i in range(self.num_threads - 1):
            if not self.master: continue
            pid = os.fork()
            if pid != 0:
                self._slavemon.notechild(pid)
            else:
                self.rank = i + 1
                self.master = False

    def _cleanup(self):
        self._barrier.abort()
        self.critical.release()
        self._StaticForLoop.abort()
        self._DynamicForLoop.abort()
        self._Ordered.abort()

    def __enter__(self):
        self.critical = Semaphore(1)
        self._errormon = ErrorMonitor() 
        self._slavemon = SlaveMonitor(self._errormon) 
        shared = sharedmem.empty((),
                dtype=[
                    ('ordered', 'intp'),
                    ('barrier', 'intp'),
                    ('dynamic', 'intp'),
                    ])

        self._barrier = Barrier(self.num_threads, shared['barrier'][...])
        self._Ordered = MetaOrdered(self, shared['ordered'] [...], Semaphore(1))
        self._StaticForLoop = MetaStaticForLoop(self) 
        self._DynamicForLoop = MetaDynamicForLoop(self, Semaphore(1), shared['dynamic'][...]) 

        for param in self._variables:
            param.beforefork(self)
        self._fork()
        for param in self._variables:
            param.afterfork(self)
        if self.master:
            self._errormon.start()
            self._slavemon.start()
            LongJump.listen(self)
        return self

    def __exitmaster__(self, type, exception, traceback):
        if type is None:
            # if any slaves raise error
            # _erromon will kill them
            # but master won't receive the LongJump signal
            self._slavemon.join()
            self._errormon.join()
            if self._errormon.message is not None:
                type, msg = pickle.loads(self._errormon.message)
                raise type(msg)
        elif type is LongJump:
            self._errormon.join()
            type, msg = pickle.loads(self._errormon.message)
            raise type(msg)
        else:
            self._slavemon.kill_all()
            self._errormon.join()

        for param in self._variables:
            if isinstance(param, Reduction):
                param.reduce(self)

    def __exit__(self, type, exception, traceback):
        try:
            # since we are already here,
            # mute further longjumps that
            # may corrupt the unprotected
            # internal implementations.
            # example
            # if Thread.join() is inproperly
            # break by a async exception
            # next call to Thread.start() may block
            LongJump.mute()
            # if LongJump is raised 
            # we simply try again
        except LongJump as e:
            LongJump.mute()

        if self.master: 
            self.__exitmaster__(type, exception, traceback)
        else:
            # put error to the pipe
            if type is not None:
                # need to make sure each (per) message 
                # won't block the pipe!
                self._errormon.slaveraise(type, exception, traceback)
            os._exit(0)
        
    def barrier(self):
        self._barrier.wait()

    def forloop(self, range, ordered=False, schedule=('static', 1)):
        """ schedule can be
            (sch, chunk) or sch;
            sch is 'static', 'dynamic' or 'guided'.

            chunk defaults to 1

            if ordered, create an ordred
        """

        if isinstance(schedule, tuple):
            schedule, chunk = schedule
        else:
            chunk = None
        if schedule == 'static':
            return self._StaticForLoop(range, ordered, chunk)
        elif schedule == 'dynamic':
            return self._DynamicForLoop(range, ordered, chunk, guided=False)
        elif schedule == 'guided':
            return self._DynamicForLoop(range, ordered, chunk, guided=True)
        else:
            raise "schedule unknown"

class Barrier:
    """ Excerpt from the Semaphore book by Downey 08 """
    def __init__(self, n, count):
        self.n = n
        self.count = count
        self.count[...] = 0
        self.mutex = Semaphore(1)
        self.turnstile = Semaphore(0)
        self.turnstile2 = Semaphore(0)

    def abort(self):
        """ ensure the master exit from Barrier """
        self.mutex.release()
        self.turnstile.release()
        self.mutex.release()
        self.turnstile2.release()

    def phase1(self):
        try:
            self.mutex.acquire()
            self.count[...] += 1
            if self.count == self.n:
                [self.turnstile.release() for i in range(self.n)]
        finally:
            self.mutex.release()
        self.turnstile.acquire()

    def phase2(self):
        try:
            self.mutex.acquire()
            self.count[...] -= 1
            if self.count == 0:
                [self.turnstile2.release() for i in range(self.n)]
        finally:
            self.mutex.release()
        self.turnstile2.acquire()
    def wait(self):
        if self.n == 0: return
        self.phase1()
        self.phase2()

def MetaDynamicForLoop(parallel, mutex, dynamiciter):
    class DynamicForLoop:
        def __init__(self, range, ordered, chunk, guided):
            self.range = range
            self.iter = numpy.empty((), dtype='intp')
            self.iter[...] = 0
            self.guided = guided
            if chunk is None: chunk = 1
            self.chunk = chunk
            if parallel.master:
                dynamiciter[...] = 0
                self._haserror = parallel._errormon.haserror
            else:
                self._haserror = lambda : False
            if ordered:
                self.ordered = parallel._Ordered(self.iter)
            else:
                self.ordered = None

            # this is important, to
            # make sure dynamiciter is updated to 0
            parallel.barrier()

        @classmethod
        def abort(self):
            mutex.release()

        def __iter__(self):
            N = len(self.range)
            while True:
                try:
                    mutex.acquire()
                    # get the unit of work
                    if not self.guided:
                        newchunk = self.chunk
                    else:
                        newchunk = (N - dynamiciter) // \
                                parallel.num_threads
                        if newchunk < self.chunk: 
                            newchunk = self.chunk
                    self.iter[...] = dynamiciter
                    dynamiciter[...] += newchunk
                finally:
                    mutex.release()

                if self.iter[...] >= N:
                    break
                for i in range(newchunk):
                    if self.iter >= N: break
                    if self.ordered is None:
                        yield self.range[self.iter]
                    else:
                        yield self.range[self.iter], self.ordered
                    self.iter[...] += 1
                if self._haserror():
                    break
            self._haserror = None
    return DynamicForLoop

def MetaStaticForLoop(parallel):
    class ForLoop:
        def __init__(self, range, ordered, chunk):
            self.start = parallel.rank * len(range) // parallel.num_threads
            self.end = (parallel.rank + 1)* len(range) // parallel.num_threads
            self.range = range
            self.iter = numpy.empty((), dtype='intp')
            self.iter[...] = 0
            if parallel.master:
                self._haserror = parallel._errormon.haserror
            else:
                self._haserror = lambda : False
            if ordered:
                self.ordered = parallel._Ordered(self.iter)
            else:
                self.ordered = None
        @classmethod
        def abort(kls):
            pass
        def __iter__(self):
            for i in range(self.start, self.end):
                if self._haserror():
                    break
                self.iter[...] = i
                if self.ordered is None:
                    yield self.range[i]
                else:
                    yield self.range[i], self.ordered
            self._haserror = None
    return ForLoop

def MetaOrdered(parallel, done, turnstile):
    """meta class for Ordered construct."""
    class Ordered:
        def __init__(self, iterref):
            if parallel.master:
                done[...] = 0
            self.iterref = iterref
            parallel.barrier()

        @classmethod
        def abort(self):
            turnstile.release()

        def __enter__(self):
            while self.iterref != done:
                pass
            turnstile.acquire()
            return self
        def __exit__(self, *args):
            done[...] += 1
            turnstile.release()
    return Ordered

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
    """ A Set of variables, used to initialize a Parallel
        section. 
        
        override before fork and afterfork in subclasses
    """
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
    """A set of shared variables.
       
       Shared(varname=value)
       if specific type is needed, eg, int8, 
       use numpy.int8.
    """
    def beforefork(self, parallel):
        self.data = sharedmem.empty((), self._dtype)
        for key, value in self._input:
            self.data[key] = value

class Private(VarSet):
    """A set of private variables"""
    def beforefork(self, parallel):
        self.data = numpy.empty((), self._dtype)
        for key, value in self._input:
            self.data[key] = value

class Reduction(VarSet):
    """A set of reduction variables
        all variables will be reduced by the same ufunc
    """
    def __init__(self, ufunc, **kwargs):
        self._ufunc = ufunc
        VarSet.__init__(self, **kwargs)

    def beforefork(self, parallel):
        self._fulldata = sharedmem.empty(parallel.num_threads, self._dtype)

    def afterfork(self, parallel):
        self.data = self._fulldata[parallel.rank]
        for key, value in self._input:
            self.data[key] = value
        VarSet.afterfork(self, parallel)

    def reduce(self, parallel):
        if parallel.master:
            for key in self:
                self[key] = self._ufunc.reduce(self._fulldata[key], axis=0)
        else:
            self.data = self._fulldata[0].view(type=Shared)

def testraiseordered():
    for schedule in ['static', 'dynamic', 'guided']:
        try:
            with Parallel(
                    Reduction(numpy.add, a=[0, 0])
                    ) as p:
                for i, ordered in p.forloop(range(20), 
                        ordered=True, schedule=schedule):
                    with ordered:
                        p.var.a += numpy.array([i, i * 10])
                        if i == 19:
                            raise ValueError('raised at i == 19')
            assert False
        except ValueError as e:
            pass

def testraisecritical():
    for schedule in ['static', 'dynamic', 'guided']:
        try:
            with Parallel(
                    Reduction(numpy.add, a=[0, 0])
                    ) as p:
                for i in p.forloop(range(20), schedule=schedule):
                    with p.critical:
                        p.var.a += numpy.array([i, i * 10])
                        if i == 19:
                            raise ValueError('raised at i == 19')
            assert False
        except ValueError as e:
            pass

def testreduction():
    for schedule in ['static', 'dynamic', 'guided']:
        with Parallel(
                Reduction(numpy.add, a=[0, 0])
                ) as p:
            for i in p.forloop(range(20), schedule=schedule) :
                p.var.a += numpy.array([i, i * 10])
        assert (p.var.a == [190, 1900]).all()

def testprivate():
    for schedule in ['static', 'dynamic', 'guided']:
        truevalue = numpy.zeros(2)
        with Parallel(
                Private(a=[0, 0])
                ) as p:
            for i in p.forloop(range(100), schedule=schedule):
                p.var.a += numpy.array([i, i * 10])
                if p.master: 
                    truevalue += numpy.array([i, i * 10])
        assert (p.var.a == truevalue).all()

def testshared():
    for schedule in ['static', 'dynamic', 'guided']:
        with Parallel(
                Shared(a=[0, 0]),
                Reduction(numpy.add, b=[0, 0])
                ) as p:
            for i in p.forloop(range(100), schedule=schedule):
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
        #time.sleep(p.rank * 0.01)
        p.barrier()
        p.barrier()
        
def testkill():
    try:
        with Parallel(
                Shared(a=[0, 0]),
                Reduction(numpy.add, b=[0, 0])
                ) as p:
        #    time.sleep(p.rank * 0.01)
            p.barrier()

            if p.rank == p.num_threads - 1:
                os.kill(os.getpid(), signal.SIGKILL)
            p.barrier()
        assert False
    except ParallelException as e:
        return
    assert False



def main():
    testkill()
    #print 'kill done'
    for i in range(100):
        print('run', i)
        testreduction()
        print('private')
        testprivate()
        print('shared')
        testshared()
        print('bairer')
        testbarrier()
        print('raisecritical')
        testraisecritical()
        print('raiseordered')
        testraiseordered()
        print('done', i)
    print('all done')

if __name__ == '__main__':
    main()
