import sys
import os

from numpy.testing import Tester
sys.path.insert(0, os.path.abspath('sharedmem'))

import sharedmem
print(sharedmem)
from sys import argv

tester = Tester()
result = tester.test(extra_argv=['-w', 'tests'] + argv[1:])
if not result.wasSuccessful():
    raise Exception("Test Failed")
