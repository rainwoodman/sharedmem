import time
import numpy
import sharedmem
import os
import signal
with sharedmem.Pool(use_threads=False) as pool:
    def work(i):
        print 'start', i
        time.sleep(numpy.random.uniform())
        #with pool.ordered:
        #    time.sleep(numpy.random.uniform())
        if i == 10:
           raise Exception("stupid exception")
        print 'end', i
    pool.map(work, range(100))
