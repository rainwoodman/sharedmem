from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="sharedmem", version="0.3",
      author="Yu Feng",
      author_email="yfeng1@andrew.cmu.edu",
      description="Easy routines for coding on sharedmem machines",
      url="http://github.com/rainwoodman/sharedmem",
      zip_safe=False,
      package_dir = {'sharedmem': 'sharedmem'},
      packages = [
        'sharedmem'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
)

