import multiprocessing
import threading

class ProcessAborted(Exception):
    def __init__(self, code):
        Exception.__init__(self, "Process killed by signal %d" % code)

class Process(multiprocessing.Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        self._target = target
        multiprocessing.Process.__init__(self, group, 
                target, name, args,
                kwargs)

        self._safelyexited = multiprocessing.RawValue('l')
        self._safelyexited.value = -1
        def guard():
            while self.is_alive() or self._safelyexited.value == -1:
                multiprocessing.Process.join(self, 1)
            if self._safelyexited.value == 0:
                raise ProcessAborted( - self.exitcode)

        self._guard = threading.Thread(target=guard)

    def run(self):
        self._safelyexited.value = 0
        multiprocessing.Process.run(self)
        self._safelyexited.value = 1

    def start(self):
        multiprocessing.Process.start(self)
        self._guard.start()

    def join(self, timeout=None):
        self._guard.join(timeout)

def main():
    import os
    import signal
    import time
    def func():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGKILL)
        
    p = Process(target=func)
    p.start()
    p.join()

#main()
