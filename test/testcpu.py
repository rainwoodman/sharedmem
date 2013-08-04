import numpy
import sharedmem
import time
with sharedmem.Pool(np=32) as pool:
    def work(i):
        print 'start', i
        time.sleep(numpy.random.uniform())
        print 'end', i
    pool.map(work, range(100))
