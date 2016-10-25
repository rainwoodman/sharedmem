from setuptools import setup
setup(name="sharedmem", version="0.3.5",
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

