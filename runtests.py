import sharedmem
import sys
import os

from numpy.testing import Tester
sys.path.insert(0, os.path.abspath('.'))

from sys import argv

tester = Tester()
tester.test(extra_argv=['-w', 'tests'] + argv[1:])

