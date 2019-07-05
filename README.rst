Dispatch your trivially parallizable jobs with sharedmem.

.. image:: https://api.travis-ci.org/rainwoodman/sharedmem.svg
    :alt: Build Status
    :target: https://travis-ci.org/rainwoodman/sharedmem/

To cite sharedmem use the DOI below

.. image:: https://zenodo.org/badge/4997909.svg
   :target: https://zenodo.org/badge/latestdoi/4997909
   
Now also supports Python 3.

- sharedmem.empty creates numpy arrays shared by child processes.

- sharedmem.MapReduce dispatches work to child processes, allowing work functions
  defined in nested scopes.

- sharedmem.MapReduce.ordered and sharedmem.MapReduce.critical implements
  the equivelant concepts as OpenMP ordered and OpenMP critical sections.

- Exceptions are properly handled, including unpicklable exceptions. Unexpected death
  of child processes (Slaves) is handled in a graceful manner.

Functions and variables are inherited from a :code:`fork` syscall and the copy-on-write
mechanism, except sharedmem variables which are writable from both child processes or the
main process.  Pickability of objects is not a concern. 

Usual limitations of :code:`fork` do apply. 
sharedmem.MapReduce is easier to use than multiprocessing.Pool, 
at the cost of not supporting Windows.

For documentation, please refer to http://rainwoodman.github.io/sharedmem .

Here we provide two simple examples to illustrate the usage:

.. code-block :: python

    """ 
        Integrate [0, ... 1.0) with rectangle rule. 
        Compare results from 
        1. direct sum of 'xdx' (filled by subprocesses)
        2. 'shmsum', cummulated by partial sums on each process
        3. sum of partial sums from each process.

    """
    xdx = sharedmem.empty(1024 * 1024 * 128, dtype='f8')
    shmsum = sharedmem.empty((), dtype='f8')

    shmsum[...] = 0.0

    with sharedmem.MapReduce() as pool:

        def work(i):
            s = slice (i, i + chunksize)
            start, end, step = s.indices(len(xdx))

            dx = 1.0 / len(xdx)

            myxdx = numpy.arange(start, end, step) \
                    * 1.0 / len(xdx) * dx

            xdx[s] = myxdx

            a = xdx[s].sum(dtype='f8')

            with pool.critical:
                shmsum[...] += a

            return i, a

        def reduce(i, a):
            # print('chunk', i, 'done', 'local sum', a)
            return a

        chunksize = 1024 * 1024

        r = pool.map(work, range(0, len(xdx), chunksize), reduce=reduce)

    assert_almost_equal(numpy.sum(r, dtype='f8'), shmsum)
    assert_almost_equal(numpy.sum(xdx, dtype='f8'), shmsum)

.. code-block :: python

    """ 
        An example word counting program. The parallelism is per line.

        In reality, the parallelism shall be at least on a file level to
        benefit from sharedmem / multiprocessing.
        
    """
    word_count = {
            'sharedmem': 0,
            'pool': 0,
            }

    with sharedmem.MapReduce() as pool:

        def work(line):
            # create a fresh local counter dictionary
            my_word_count = dict([(word, 0) for word in word_count])

            for word in line.replace('.', ' ').split():
                if word in word_count:
                    my_word_count[word] += 1

            return my_word_count

        def reduce(her_word_count):
            for word in word_count:
                word_count[word] += her_word_count[word]

        pool.map(work, file(__file__, 'r').readlines(), reduce=reduce)

        parallel_result = dict(word_count)

        # establish the ground truth from the sequential counter
        sharedmem.set_debug(True)

        for word in word_count:
            word_count[word] = 0

        pool.map(work, file(__file__, 'r').readlines(), reduce=reduce)
        sharedmem.set_debug(False)

    for word in word_count:
        assert word_count[word] == parallel_result[word]


Segfault when work function returns raw pointers
------------------------------------------------

Although the global variables are delivered via copy-on-write fork,
sharedmem relies on python's pickle module to send and recieve the
return value of 'work' functions.

As a consequence, if the underlying library used by the work function
returns objects that are not pickle friendly,
then we will receive a corrupted object on the master process.


This can happen,
for example if the underlyihng library returns an object that stores a raw
pointer as an attribute. After unpickling the result on a new process, the raw
pointer will point to an undefined memory region, and the master process will
segfault as a result.

It is not as exotic as it sounds. We ran into this issue when interfacing sharedmem with
cosmosis, which stores a raw pointer as an attribute:

https://bitbucket.org/joezuntz/cosmosis/src/2a9d3197852f900555ee9c72784604f4a1773ee1/cosmosis/datablock/cosmosis_py/block.py#lines-78

The solution is to unpack the result of the work function, down to low level objects that are pickle
friendly, and return those instead of the unfriendly high level object.

