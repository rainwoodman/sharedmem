import numpy

__all__ = ['blockarray']

class block(numpy.ndarray):
    def __new__(cls, colname, array, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        array = numpy.array(array, copy=False)
        assert array.shape[:len(shape)] == shape
        self = array.view(type=cls)
        self.colname = colname
        self.nativeshape = shape
        self.itemshape = array.shape[len(shape):]
        return self

    def __repr__(self):
        return 'block(%s, ..., %s)' % (repr(self.colname), repr(self.shape))
    def __str__(self):
        return '%s:%s' % (str(self.colname), str(self.view(type=numpy.ndarray)))

class blockarray(object):
    """ an array object where data is saved in columns.
        data in the same column are saved in a ndarray """

    def __init__(self, shape, cols=[]):
        """cols is a list of (colname, values), shape has to be 1D."""
        self._arrays = {}
        self._names = []
        self.shape = shape
        for colname, value in cols:
            self.set(colname, value)

        self._dtype = None

    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, value):
        try:
            self._shape = tuple(value)
        except:
            self._shape = tuple([value])
        assert len(self._shape) == 1

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = numpy.dtype([
            (name, (self._arrays[name].dtype, 
                self._arrays[name].itemshape)) for name in
                self._names])
        return self._dtype

    def __len__(self):
        return self.shape[0]

    def set(self, colname, value):
        b = block(colname, value, self.shape)
        self._names.append(colname)
        self._arrays[colname] = b
        self._dtype = None

    def remove(self, colname):
        self._names.remove(colname)
        del self._arrays[colname]
        self._dtype = None

    def get(self, colname):
        if colname not in self._names:
            raise IndexError("index `%s' not found" % colname)
        return self._arrays[colname]

    def extract(self, condition):
        buf = numpy.empty(condition.sum(), dtype=self.dtype)
        for colname in self._names:
            buf[colname] = self[colname][condition]
        return buf

    def take(self, indices):
        buf = numpy.empty(indices.shape, dtype=self.dtype)
        for colname in self._names:
            buf[colname] = self[colname][indices]
        return buf

    def place(self, condition, values):
        for colname in self._names:
            self[colname][condition] = values[colname]

    def put(self, indices, values):
        for colname in self._names:
            self[colname][indices] = values[colname]

    def append(self, addition):
        return blockarray(
            len(self) + len(addition),
            [
                (colname, 
                numpy.append(self[colname], addition[colname], axis=0)
                ) 
                for colname in self._names    
            ])

    def sort(self, orderby):
        assert not isinstance(orderby, (tuple, list))
        # cannot do multiple keys
        arg = self[orderby].argsort()
        self.permute(arg)

    def permute(self, arg):
        for colname in self._names:
            self[colname][...] = self[colname][arg]

    def __getitem__(self, index):
        if isinstance(index, basestring):
            return self.get(index)
        if isinstance(index, (tuple, list)):
            index = numpy.array(index, dtype='intp')
        if isinstance(index, numpy.ndarray):
            if index.dtype.char == '?':
                return self.extract(index)
            else:
                return self.take(index)
        if isinstance(index, slice):
            start, stop, stride = index.indices(len(self))
            return blockarray(
                    (stop - start) // stride,
                    [(colname, self[colname][index]) for colname
                        in self._names])

        item = numpy.empty((), self.dtype)
        for colname in self._names:
            item[colname] = self[colname][index]
        return item

    def __setitem__(self, index, value):
        if isinstance(index, basestring):
            return self.set(index, value)
        if isinstance(index, (tuple, list)):
            index = numpy.array(index, dtype='intp')
        if isinstance(index, numpy.ndarray):
            if index.dtype.char == '?':
                return self.place(index, value)
            else:
                return self.put(index, value)
        for colname in self._names:
            self[colname][index] = value[colname]
    def __repr__(self):
        return 'blockarray(%d, ... %s) '% (len(self), repr(self.dtype))
    def __str__(self):
        return 'blockarray(%d, ... %s) '% (len(self), repr(self.dtype))

if __name__ == '__main__':
    a = numpy.array([10, 20, 30])
    b = numpy.ones(3, dtype=('f8', 2))
    print block('abc', a, 3)
    t = blockarray(3)
    t['a'] = a
    t['b'] = b
    print numpy.append(t, t)
