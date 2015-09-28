import numpy
import sharedmem
import time

from numpy.testing import (assert_equal, assert_array_equal,
    assert_almost_equal, assert_array_almost_equal, assert_, run_module_suite)
import sys

def test_create():
    a = sharedmem.empty(100, dtype='f8')
    l = list(a)
    b = sharedmem.empty_like(a)
    assert b.shape == a.shape
    b = sharedmem.empty_like(l)
    assert b.shape == a.shape
    c = sharedmem.full(100, 1.0, dtype='f8')
    assert c.shape == a.shape
    c = sharedmem.full_like(a, 1.0)
    assert c.shape == a.shape

def run_idle(pool):
    def work(i):
        time.sleep(0.4)

    with pool:
        now = time.time()
        pool.map(work, range(pool.np))

    return time.time() - now

def test_parallel_process():
    pool = sharedmem.MapReduce()
    assert run_idle(pool) < 1.0

def test_parallel_thread():
    pool = sharedmem.MapReduceByThread() 
    assert run_idle(pool) < 1.0

from sharedmem import background


def test_background():
    def function1():
        time.sleep(2)
        return True

    re = background(function1)
    now = time.time()
    assert re.wait() == True
    assert int(time.time() - now + 0.5) == 2

def test_background_raise():
    def function2():
        raise Exception('test exception')
        time.sleep(2)
    re = background(function2)
    now = time.time()
    try:
        assert re.wait() == True
    except Exception as e:
        return 

    raise AssertionError("Shall not reach here")

def test_killed():
    import os
    import signal

    with sharedmem.MapReduce() as pool:
        def work(i):
            time.sleep(0.1 * numpy.random.uniform())
            if i == 10:
               os.kill(os.getpid(), signal.SIGKILL)
        try:
            pool.map(work, range(100))
        except sharedmem.SlaveException:
            return

    raise AssertionError("Shall not reach here")

class UnpicklableException(Exception):
    def __reduce__(self):
        raise Exception("This pickle is not supposed to be pickled")

import warnings
def test_unpicklable_raise():
    with sharedmem.MapReduce() as pool:
        def work(i):
            time.sleep(0.1 * numpy.random.uniform())
            if i == 10:
                raise UnpicklableException("Raise an exception")
        try:
            with warnings.catch_warnings(record=True) as w:
                pool.map(work, range(100))
            # except an warning here
            assert len(w) == 1 
        except Exception as e:
            assert not isinstance(e.reason, UnpicklableException) 
            return
    raise AssertionError("Shall not reach here")

class PicklableException(Exception):
    pass

def test_picklable_raise():
    with sharedmem.MapReduce() as pool:
        def work(i):
            time.sleep(0.1 * numpy.random.uniform())
            if i == 10:
                raise PicklableException("Raise an exception")
        try:
            pool.map(work, range(100))
        except sharedmem.SlaveException as e:
            assert isinstance(e.reason, PicklableException)
            return
    raise AssertionError("Shall not reach here")

def test_memory_pickle():
    import pickle
    a = sharedmem.empty(100)
    a[:] = range(100)
    s = pickle.dumps(a)
    b = pickle.loads(s)

    assert isinstance(b, type(a))

    b[:] += 10
    assert (a == b).all()

def test_memory_type():
    a = sharedmem.empty(100)
    b = sharedmem.empty(100)
    assert isinstance(b, type(a))

    assert not isinstance(a + 10, type(a))
    assert not isinstance(numpy.sum(a), type(a))
    assert not isinstance(a + b, type(a))
    assert not isinstance(a * b, type(a))

def test_ordered():
    t = sharedmem.empty(800)
    with sharedmem.MapReduce(np=32) as pool:
        def work(i):
            time.sleep(0.1 * numpy.random.uniform())
            with pool.ordered:
                t[i] = time.time()
        pool.map(work, range(800))

        # without ordered, the time is ordered
        assert (t[1:] > t[:-1]).all()

        def work(i):
            time.sleep(0.1 * numpy.random.uniform())
            t[i] = time.time()
        pool.map(work, range(800))
        # without ordered, the ordering is messy
        assert not (t[1:] > t[:-1]).all()

def test_critical():
    t = sharedmem.empty(1, dtype='i8')
    t[:] = 0
    # FIXME: if the system has one core then this will never fail,
    # even if the critical section is not 
    with sharedmem.MapReduce(np=8) as pool:
        def work(i):
            with pool.critical:
                t[:] = 1
                if i != 30:
                    time.sleep(0.01)
                assert t[:] == 1
                t[:] = 0

        pool.map(work, range(16))

        def work(i):
            t[:] = 1
            if i != 30:
                time.sleep(0.01)
            assert t[:] == 1
            t[:] = 0

        try:
            pool.map(work, range(16))
        except sharedmem.SlaveException as e:
            assert isinstance(e.reason, AssertionError)
            return 
    raise AssertionError("Shall not reach here.")

if __name__ == "__main__":
    import sys
    run_module_suite()
