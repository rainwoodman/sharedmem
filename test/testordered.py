import sharedmem
import time
import numpy
with sharedmem.Pool(use_threads=False) as pool:
    def work(i):
        print 'start', i
        time.sleep(numpy.random.uniform())
        #with pool.ordered:
        #    time.sleep(numpy.random.uniform())
        print 'end', i
    pool.map(work, range(100))

