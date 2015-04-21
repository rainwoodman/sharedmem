import numpy
__all__ = ['MultiPartStream']

class MultiPartStream:
    def __init__(self, files, fetch):
        """ fetch is a method that takes 
              fetch(filename)

            it shall read in the content of filename in ndarray
            if fetch(None), returns an empty size=0 ndarray
            with the same spec of the correct file


            pool needs to support copy() and indexing.
        """
        self.files = files
        self.fetch = fetch
        self.pool = fetch(None)

    def iter(self, blocksize):
        try: 
            while True:
                if blocksize is None:
                    self.cur = self.files.next()
                    yield self.fetch(self.cur)
                else:
                    yield self.read(blocksize)
        except StopIteration:
            return

    def read(self, n):
        """ return at most n array items, move the cursor. 
        """
        while len(self.pool) < n:
            self.cur = self.files.next()
            self.pool = numpy.append(self.pool,
                    self.fetch(self.cur), axis=0)

        rt = self.pool[:n]
        if n == len(self.pool):
            self.pool = self.fetch(None)
        else:
            self.pool = self.pool[n:]
        return rt
