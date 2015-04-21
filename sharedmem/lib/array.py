import numpy
import sharedmem
#from .. import core as sharedmem

__all__ = ['pufunc']

_funcs = [
'minimum', 'maximum',
'add', 'subtract',
'multiply',
'divide',
'sin',
'cos',
'tan',
'exp',
'log',
'log10',
]
_funcs2 = [ 
        ('searchsorted', [1], numpy.dtype('intp'), None),
        ('digitize', [0], numpy.dtype('intp'), None),
        ('bincount', [0], numpy.dtype('intp'), numpy.add),
        ('isnan', [0], numpy.dtype('bool'), None),
        ('isinf', [0], numpy.dtype('bool'), None),
    ]


class pufunc(object):
    def __init__(self, func, ins=None, outdtype=None, altreduce=None):
        """ if func is not ufunc, a bit complicated:
            ins tells which positional argument will be striped
            after done, reducefunc is called on the results
        """
        if isinstance(func, numpy.ufunc):
            self.ufunc = func
            self.nin = func.nin
            self.ins = (0, 1, 2, 3)[:func.nin]
            self.nout = func.nout
            self.outdtype = None
            self.altreduce = None
        else:
            self.ufunc = func
            self.nin = len(ins)
            self.ins = ins
            self.nout = 1
            self.outdtype = outdtype
            self.altreduce = altreduce

        self.__doc__ = func.__doc__
        if self.nout != 1:
            raise TypeError("only support 1 out ufunc")

    def reduce(self, a, axis=0, dtype=None, chunksize=1024 * 1024):
        rt = [None]
        if axis != 0:
            a = numpy.rollaxis(a, axis)

        with sharedmem.MapReduce() as pool:
            def work(i):
                sl = slice(i, i+chunksize)
                if len(a[sl]) == 0:
                    return self.ufunc.identity
                else:
                    return self.ufunc.reduce(a[sl], axis, dtype)
            def reduce(r):
                if rt[0] is None:
                    rt[0] = r
                elif r is None:
                    pass
                else:
                    rt[0] = self.ufunc(rt[0], r, dtype=dtype)
            pool.map(work, 
                    range(0, len(a), chunksize),
                    reduce=reduce)
        return rt[0]

    def __call__(self, *args, **kwargs):
        return self.call(list(args), **kwargs)

    def call(self, args, axis=0, out=None, chunksize=1024 * 1024, **kwargs):
        """ axis is the axis to chop it off.
            if self.altreduce is set, the results will
            be reduced with altreduce and returned
            otherwise will be saved to out, then return out.
        """
        if self.altreduce is not None:
            ret = [None]
        else:
            if out is None :
                if self.outdtype is not None:
                    dtype = self.outdtype
                else:
                    try:
                        dtype = numpy.result_type(*[args[i] for i in self.ins] * 2)
                    except:
                        dtype = None
                out = sharedmem.empty(
                        numpy.broadcast(*[args[i] for i in self.ins] * 2).shape,
                        dtype=dtype)
        if axis != 0:
            for i in self.ins:
                args[i] = numpy.rollaxis(args[i], axis)
            out = numpy.rollaxis(out, axis)
        size = numpy.max([len(args[i]) for i in self.ins])
        with sharedmem.MapReduce() as pool:
            def work(i):
                sl = slice(i, i+chunksize)
                myargs = args[:]
                for j in self.ins:
                    try: 
                        tmp = myargs[j][sl]
                        a, b, c = sl.indices(len(args[j]))
                        myargs[j] = tmp
                    except Exception as e:
                        print tmp
                        print j, e
                        pass
                if b == a: return None
                rt = self.ufunc(*myargs, **kwargs)
                if self.altreduce is not None:
                    return rt
                else:
                    out[sl] = rt
            def reduce(rt):
                if self.altreduce is None:
                    return
                if ret[0] is None:
                    ret[0] = rt
                elif rt is not None:
                    ret[0] = self.altreduce(ret[0], rt)

            pool.map(work, range(0, size, chunksize), reduce=reduce)

        if self.altreduce is None:
            if axis != 0:
                out = numpy.rollaxis(out, 0, axis + 1)
            return out                
        else:
            return ret[0]

for _f in _funcs:
    globals()[_f] = pufunc(getattr(numpy, _f))
    __all__ .append(_f)
for _f,_ins,_out,_altr in _funcs2:
    globals()[_f] = pufunc(getattr(numpy, _f), _ins, _out, _altr)
    __all__ .append(_f)

def argsort(ar):
    min = minimum.reduce(ar)
    max = maximum.reduce(ar)
    nchunk = sharedmem.cpu_count() * 2
    #bins = numpy.linspace(min, max, nchunk, endpoint=True)
    step = 1.0 * (max - min) / nchunk
    bins = numpy.array(
            1.0 * numpy.arange(nchunk + 1) * (max - min) / nchunk + min,
            min.dtype)

    dig = digitize(ar, bins)
    binlength = bincount(dig, minlength=len(bins) + 1)
    binoffset = numpy.cumsum(binlength)
    out = sharedmem.empty(len(ar), dtype='intp')

    with sharedmem.MapReduce() as pool:
        def work(i):
            # we can do this a lot faster
            # but already having pretty good speed.
            ind = numpy.nonzero(dig == i + 1)[0]
            myar = ar[ind]
            out[binoffset[i]:binoffset[i+1]] = ind[myar.argsort()]
        pool.map(work, range(nchunk)) 

    return out

class packarray(numpy.ndarray):
  """ A packarray packs/copies several arrays into the same memory chunk.

      It feels like a list of arrays, but because the memory chunk is continuous,
      
      arithmatic operations are easier to use(via packarray.A)
  """
  def __new__(cls, array, start=None, end=None):
    """ if end is none, start contains the sizes. 
        if start is also none, array is a list of arrays to concatenate
    """
    self = array.view(type=cls)
    if end is None and start is None:
      start = numpy.array([len(arr) for arr in array], dtype='intp')
      array = numpy.concatenate(array)
    if end is None:
      sizes = start
      self.start = numpy.zeros(shape=len(sizes), dtype='intp')
      self.end = numpy.zeros(shape=len(sizes), dtype='intp')
      self.end[:] = sizes.cumsum()
      self.start[1:] = self.end[:-1]
    else:
      self.start = start
      self.end = end
    self.A = array
    return self
  @classmethod
  def adapt(cls, source, template):
    """ adapt source to a packarray according to the layout of template """
    if not isinstance(template, packarray):
      raise TypeError('template must be a packarray')
    return cls(source, template.start, template.end)

  def __repr__(self):
    return 'packarray: %s, start=%s, end=%s' % \
          (repr(self.A), 
           repr(self.start), repr(self.end))
  def __str__(self):
    return repr(self)

  def copy(self):
    return packarray(self.A.copy(), self.start, self.end)

  def compress(self, mask):
    count = self.end - self.start
    realmask = numpy.repeat(mask, count)
    return packarray(self.A[realmask], self.start[mask], self.end[mask])

  def __getitem__(self, index):
    if isinstance(index, basestring):
      return packarray(self.A[index], self.end - self.start)

    if isinstance(index, slice) :
      start, end, step = index.indices(len(self))
      if step == 1:
        return packarray(self.A[self.start[start]:self.end[end]],
            self.start[start:end] - self.start[start],
            self.end[start:end] - self.start[start])

    if isinstance(index, (list, numpy.ndarray)):
      return packarray(self.A, self.start[index], self.end[index])

    if numpy.isscalar(index):
      start, end = self.start[index], self.end[index]
      if end > start: return self.A[start:end]
      else: return numpy.empty(0, dtype=self.A.dtype)
    raise IndexError('unsupported index type %s' % type(index))

  def __len__(self):
    return len(self.start)

  def __iter__(self):
    for i in range(len(self.start)):
      yield self[i]

  def __reduce__(self):
    return packarray, (self.A, self.end - self.start)

  def __array_wrap__(self, outarr, context=None):
    return packarray.adapt(outarr.view(numpy.ndarray), self)
#    return numpy.ndarray.__array_wrap__(self.view(numpy.ndarray), outarr, context)
