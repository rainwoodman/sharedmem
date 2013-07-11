import sharedmem
import time
import numpy
with sharedmem.Pool(use_threads=True) as pool:
    def work(i):
        time.sleep(numpy.random.uniform())
        with pool.ordered:
    #        time.sleep(numpy.random.uniform())
            print 'ordered', i
    pool.map(work, range(100))

