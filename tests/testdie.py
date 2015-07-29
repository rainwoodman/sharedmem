import time
import numpy
import sharedmem
import os
import signal

def test():
    with sharedmem.MapReduce() as pool:
        def work(i):
            print('start', i)
            time.sleep(numpy.random.uniform())
            #with pool.ordered:
            #    time.sleep(numpy.random.uniform())
            if i == 10:
               os.kill(os.getpid(), signal.SIGKILL)
            print('end', i)
        try:
            pool.map(work, range(100))
        except sharedmem.SlaveException:
            return
    assert False
test()
