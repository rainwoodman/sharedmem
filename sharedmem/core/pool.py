from .mapreduce import MapReduce
from . import backends
__all__ = ['Pool', 'TPool']

class Pool(MapReduce):
    def __init__(self, np=None):
        MapReduce.__init__(self, backend=backends.ProcessBackend)

class TPool(MapReduce):
    def __init__(self, np=None):
        MapReduce.__init__(self, backend=backends.ThreadBackend)

