from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="sharedmem", version="0.1",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="Easy routines for coding on sharedmem machines",
      url="http://github.com/rainwoodman/sharedmem",
      zip_safe=False,
      package_dir = {'sharedmem': 'src'},
      packages = [
        'sharedmem'
      ],
      requires=['numpy'],
      ext_modules = [
        Extension('sharedmem.' + name, 
             [ 'src/' + name.replace('.', '/') + '.c',],
             extra_compile_args=['-O3'],
             libraries=[],
             include_dirs=[get_include()],
             depends = extra
        ) for name, extra in [
         ('listtools', []),
         ('_mergesort', []),
        ]
      ])

