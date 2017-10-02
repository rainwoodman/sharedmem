# pop the current directory from search path
# python interpreter adds this to a top level script
# but we will likely have a name conflict (runtests.py .vs runtests package)
import sys; sys.path.pop(0)
from runtests import Tester
import os.path

tester = Tester(os.path.abspath(__file__), "sharedmem")

tester.main(sys.argv[1:])
