import backends
import traceback
__all__ = ['background']

class background(object):
    """ to run a function in async with a process.

        def function(*args, **kwargs):
            pass

        bg = background(function, *args, **kwargs)

        rt = bg.wait()
    """
    def __init__(self, function, backend=backends.ProcessBackend, *args, **kwargs):

        self.result = backend.QueueFactory(1)
        self.slave = backend.SlaveFactory(target=self.closure, 
                args=(function, args, kwargs, self.result))
        self.slave.start()

    def closure(self, function, args, kwargs, result):
        try:
            rt = function(*args, **kwargs)
        except Exception as e:
            result.put((e, traceback.format_exc()))
        else:
            result.put((None, rt))

    def wait(self):
        e, r = self.result.get()
        self.slave.join()
        self.slave = None
        self.result = None
        if isinstance(e, Exception):
            raise backends.SlaveException(e, r)
        return r
