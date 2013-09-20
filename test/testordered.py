import sharedmem
import time
import numpy
with sharedmem.Pool() as pool:
    t = sharedmem.empty(100)
    def work(i):
        time.sleep(numpy.random.uniform())
        with pool.ordered:
    #        time.sleep(numpy.random.uniform())
            t[i] = time.time()
            print 'ordered', i
    pool.map(work, range(100))
    assert (t[1:] > t[:-1]).all()
