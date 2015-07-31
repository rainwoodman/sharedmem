Dispatch your trivially parallizable jobs with sharedmem.

Now also supports Python 3.

.. image:: https://api.travis-ci.org/rainwoodman/sharedmem.svg
    :alt: Build Status

- sharedmem.empty creates numpy arrays to child processes.

- sharedmem.MapReduce dispatches work to child processes.

- sharedmem.MapReduce.ordered and sharedmem.MapReduce.critical provides
  the equivlant concept of OpenMP ordered and OpenMP critical sections.

Functions and variables are inherited from a `fork` and copy-on-write. 
Pickability is not a concern. 

Easier to use than multiprocessing.Pool, at the cost of not supporting Windows.

For documentation, please refer to http://rainwoodman.github.io/sharedmem .

>>>
>>> input = numpy.arange(1024 * 1024 * 128, dtype='f8')
>>> output = sharedmem.empty(1024 * 1024 * 128, dtype='f8')
>>> with MapReduce() as pool:
>>>    chunksize = 1024 * 1024
>>>    def work(i):
>>>        s = slice (i, i + chunksize)
>>>        output[s] = input[s]
>>>        return i, sum(input[s])
>>>    def reduce(i, r):
>>>        print('chunk', i, 'done')
>>>        return r
>>>    r = pool.map(work, range(0, len(input), chunksize), reduce=reduce)
>>> print numpy.sum(r)
>>>


