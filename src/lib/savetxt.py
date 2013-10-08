"""
    an alternative to numpy.savetxt. 
    this is sequential, but writes a nice header to text files.
    also supports nested datatypes.

"""
import numpy
import re
import base64
import pickle
import shlex
def savetxt2(fname, X, delimiter=' ', newline='\n', comment_character='#',
        fmt={}):

    """ format of header:

        ID [type]:name(index) .... * number of items

        name of nested dtype elements are split by .
    """
    prefixfmt = {}
    for key in fmt:
            prefixfmt[key] = fmt[key]

    olddtype = X.dtype
    X = X.view(dtype=flatten_dtype(X.dtype))
    dtype = X.dtype
    header = _mkheader(dtype)
    fmtstr = _mkfmtstr(dtype, prefixfmt, delimiter, _default_fmt)

    if hasattr(fname, 'write'):
        fh = fname
        cleanup = lambda : None
    else:
        fh = file(fh, 'w+')
        cleanup = lambda : fh.close()
    try:
        fh.write (comment_character)
        fh.write ('!')
        fh.write (header)
        fh.write (delimiter)
        fh.write ('*%d' % len(X))
        fh.write(newline)
        fh.write (comment_character)
        fh.write ('?')
        fh.write (base64.b64encode(pickle.dumps(olddtype)))
        fh.write (newline)
        for row in X:
            fh.write(fmtstr % tuple(row))
            fh.write(newline)

        if hasattr(fh, 'flush'):
            fh.flush()
    finally:
        cleanup()
    
def loadtxt2(fname, delimiter=' ', newline='\n', comment_character='#',
        skiplines=0):
    """ Known issues delimiter and newline is not respected. 
        string quotation with space is broken.
    """
    dtypert = [None, None, None]
    def fileiter(fh):
        dtype = None
        flatten = None
        converters = []
        buf = None
        for lineno, line in enumerate(fh):
            if lineno < skiplines: continue
            if line[0] in comment_character:
                if line[1] == '?':
                    dtype = pickle.loads(base64.b64decode(line[2:]))
                    dtypert[0] = dtype
                    flatten = flatten_dtype(dtype)
                    dtypert[1] = flatten
                    dtypert[2] = numpy.dtype([('a', (numpy.int8,
                        flatten.itemsize))])
                    buf = numpy.empty((), dtype=flatten)
                    converters = [_default_conv[flatten[name].char] for name in flatten.names]
                    yield None
                continue
            for word, c, name in zip(line.split(), converters, flatten.names):
                buf[name] = c(word)
            buf2 = buf.copy().view(dtype=dtypert[2])
            yield buf2

    if hasattr(fname, 'read'):
        fh = fname
        cleanup = lambda : None
    else:
        fh = file(fh, 'r')
        cleanup = lambda : fh.close()
    try:
        iter = fileiter(fh)
        iter.next()
        return numpy.fromiter(iter, dtype=dtypert[2]).view(dtype=dtypert[0]) 
    finally:
        cleanup()

def test():
    from StringIO import StringIO
    d = numpy.dtype(
        [
           ('a', 'i4'),
           ('b', ([('c', 'S10')], 2)),
           ('d', numpy.dtype([('e', ('i4', 5)), ('f', 'S2')]))
           ])
    a = numpy.zeros(2, d)
    a['d']['e'][0] = [1, 2, 3, 4, 5]
    a['d']['e'][1] = [1, 2, 3, 4, 5]
    s = StringIO()
    savetxt2(s, a, fmt=dict([('a', '0x%.8X'), ('d.e', '%.8d')]))
    print s.getvalue()
    print loadtxt2(StringIO(s.getvalue()))

def _mkheader(dtype):
    return ' '.join(
            ['%d[%s]:%s' % (i, dtype[name].str, name) for i, name in
            enumerate(dtype.names)])

def _mkfmtstr(dtype, prefixfmt, delimiter, defaultfmt):
    l = []
    for name in dtype.names:
        val = None
        for key in prefixfmt:
            if name.startswith(key):
                val = prefixfmt[key]
                break
        if val is None:
            val = defaultfmt[dtype[name].char]
        l.append(val)
    return delimiter.join(l)
    
def _mkvalrow(dtype, row):
    vallist = []
    if dtype.names is None and dtype.base == dtype:
        if len(dtype.shape) == 0:
            vallist.append(row)
        else:
            for i in numpy.ndindex(dtype.shape):
                vallist.append(row[i])
    elif dtype.names is None:
        for i in numpy.ndindex(dtype.shape):
            var = _mkvalrow(dtype.base, row[i])
            vallist += var
    else:
        for field in dtype.names:
            var = _mkvalrow(dtype[field], row[field])
            vallist += var

    return vallist

def _psvalrow(dtype, row, vallist):
    if dtype.names is None and dtype.base == dtype:
        if len(dtype.shape) == 0:
            row[...] = dtype.type(vallist[0])
            vallist = vallist[1:]
        else:
            for i in numpy.ndindex(dtype.shape):
                row[i][...] = dtype.type(vallist[0])
                vallist = vallist[1:]
    elif dtype.names is None:
        for i in numpy.ndindex(dtype.shape):
            vallist = _psvalrow(dtype.base, row[i], vallist)
    else:
        for field in dtype.names:
            vallist = _psvalrow(dtype[field], row[field][...], vallist)

    return vallist

def simplerepr(i):
    if len(i) == 0:
        return ''
    if len(i) == 1:
        return '(' + str(i[0]) + ')'
    return '(' + str(i) + ')'

def flatten_dtype(dtype, _next=None):
    """ Unpack a structured data-type.  """
    types = []
    if _next is None: 
        _next = [0, '']
        primary = True
    else:
        primary = False

    prefix = _next[1]

    if dtype.names is None:
        for i in numpy.ndindex(dtype.shape):
            if dtype.base == dtype:
                types.append(('%s%s' % (prefix, simplerepr(i)), dtype))
                _next[0] += 1
            else:
                _next[1] = '%s%s' % (prefix, simplerepr(i))
                types.extend(flatten_dtype(dtype.base, _next))
    else:
        for field in dtype.names:
            typ_fields = dtype.fields[field]
            if len(prefix) > 0:
                _next[1] = prefix + '.' + field
            else:
                _next[1] = '' + field
            flat_dt = flatten_dtype(typ_fields[0], _next)
            types.extend(flat_dt)

    _next[1] = prefix
    if primary:
        return numpy.dtype(types)
    else:
        return types
    
_default_fmt = {
        'f': '%g'        ,
        'i': '%d'        ,
        'I': '%d'        ,
        'L': '%d'        ,
        'S': '"%s"'        ,
}
_default_conv = {
        'f': float        ,
        'i': lambda x: long(x, base=0),
        'L': lambda x: long(x, base=0),
        'I': lambda x: long(x, base=0),
        'S': lambda x: str(x[1:-1]),
}

if __name__ == '__main__':
    test()

