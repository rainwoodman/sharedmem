from sharedmem import background
import time

def function1():
    time.sleep(3)
    return True

def function2():
    raise Exception('test exception')
    time.sleep(3)

def test1():
    re = background(function1)
    now = time.time()
    assert re.wait() == True
    assert int(time.time() - now + 0.5) == 3

def test2():
    re = background(function2)
    now = time.time()
    try:
        assert re.wait() == True
    except Exception as e:
        print e

test1()
test2()
