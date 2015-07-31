from numpy.distutils.core import setup, Extension
from numpy import get_include
setup(name="sharedmem", version="0.3",
      author="Yu Feng",
      author_email="rainwoodman@gmail.com",
      description="Dispatch your trivially parallizable jobs with sharedmem. ",
      url="http://github.com/rainwoodman/sharedmem",
      zip_safe=False,
      package_dir = {'sharedmem': 'sharedmem'},
      packages = [
        'sharedmem'
      ],
      requires=['numpy'],
      install_requires=['numpy'],
)

