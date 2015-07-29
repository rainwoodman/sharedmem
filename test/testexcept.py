import time
import numpy
import sharedmem
import os
import signal

class MyException(Exception):
    def __reduce__(self):
        raise Exception("Cannot pickle this thing")
def test():
    with sharedmem.MapReduce() as pool:
        def work(i):
            print( 'start', i)
            time.sleep(numpy.random.uniform())
            #with pool.ordered:
            #    time.sleep(numpy.random.uniform())
            if i == 10:
                raise MyException("Raise an exception")
            print('end', i)
        try:
            pool.map(work, range(100))
        except Exception as e:
            print('caught', type(e))
            return
    assert False

test()
