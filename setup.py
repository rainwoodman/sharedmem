from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="sharedmem", version="0.2",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="Easy routines for coding on sharedmem machines",
      url="http://github.com/rainwoodman/sharedmem",
      download_url="http://web.phys.cmu.edu/~yfeng1/gaepsi/sharedmem-0.2.tar.gz",
      zip_safe=False,
      package_dir = {'sharedmem': 'sharedmem'},
      packages = [
        'sharedmem', 'sharedmem.core' #, 'sharedmem.lib'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
#      ext_modules = [
#        Extension('sharedmem.' + name, 
#             [ 'sharedmem/' + name.replace('.', '/') + '.c',],
#             extra_compile_args=['-O3'],
#             libraries=[],
#             include_dirs=[get_include()],
#             depends = extra
#        ) for name, extra in [
#         ('lib._mergesort', []),
        #]
#      ]
)

