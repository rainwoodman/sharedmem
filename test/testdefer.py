from sharedmem import background
import time
def function():
    print 'defer finished'
    raise Exception('bad')
    time.sleep(3)
re = background(function)
print 'defer created'

print 'main done'
re.wait()
