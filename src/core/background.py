import numpy
import backends
import Queue as queue
import traceback
__all__ = ['background']

class background(object):
    def __init__(self, function, *args, **kwargs):
        backend=backends.ProcessBackend
        self.result = backend.QueueFactory(1)
        # reference parameters
        self.args = args
        self.kwargs = kwargs
        def closure():
            try:
                rt = function(*self.args, **self.kwargs)
            except Exception as e:
                self.result.put((e, traceback.format_exc()))
            else:
                self.result.put((None, rt))
        self.slave = backend.SlaveFactory(target=closure)
        self.slave.start()

    def wait(self):
        e, r = self.result.get()
        self.slave.join()
        self.slave = None
        self.result = None
        self.args = None
        self.kwargs = None
        if isinstance(e, Exception):
            raise Exception(str(e) + "\ntraceback:\n" + r)
        return r
