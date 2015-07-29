import numpy
import sharedmem
import time

with sharedmem.Pool() as pool:
    def work(i):
        time.sleep(0.2)
    now = time.time()
    pool.map(work, range(pool.np * 16))
    print('Pool took', time.time() - now)

with sharedmem.TPool() as pool:
    def work(i):
        time.sleep(0.2)
    now = time.time()
    pool.map(work, range(pool.np * 16))
    print('TPool took', time.time() - now)
