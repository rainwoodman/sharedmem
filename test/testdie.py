import time
import numpy
import sharedmem
import os
import signal

with sharedmem.Pool() as pool:
    def work(i):
        print 'start', i
        time.sleep(numpy.random.uniform())
        #with pool.ordered:
        #    time.sleep(numpy.random.uniform())
        if i == 10:
           os.kill(os.getpid(), signal.SIGKILL)
        print 'end', i
    pool.map(work, range(100))
