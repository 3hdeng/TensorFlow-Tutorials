# from time import clock
import time

x= divmod(5123, 1000)
print x
y=x + (123,)
print y

t=5.123
x1=reduce(lambda arr,b : divmod(arr[0],b) + arr[1:],
            [(t*1000,),1000,60,60])
print x1

# t=time.clock()
t=time.time()

def secondsToStr(t):
    return "%d:%02d:%02d.%03d" % \
        reduce(lambda ll,b : divmod(ll[0],b) + ll[1:],
            [(t*1000,),1000,60,60])

print secondsToStr(t/1000)


def mysec2Str(t):
   m,s=divmod(t,60)
   print m,s   
   h,m=divmod(m,60)
   print h,m
   return "{0}:{1:02d}:{2:02d}".format(int(h),int(m),int(s))

#https://pyformat.info/

print mysec2Str(t/1000)

"""
time.clock()
This method returns the current processor time as a floating point number expressed in seconds

reduce(function, iterable[, initializer])
Apply function of two arguments cumulatively to the items of iterable, from left to right, so as to reduce the iterable to a single value. For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates ((((1+2)+3)+4)+5). The left argument, x, is the accumulated value and the right argument, y, is the update value from the iterable. If the optional initializer is present, it is placed before the items of the iterable in the calculation, and serves as a default when the iterable is empty. If initializer is not given and iterable contains only one item, the first item is returned. Roughly equivalent to:

def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        try:
            initializer = next(it)
        except StopIteration:
            raise TypeError('reduce() of empty sequence with no initial value')
    accum_value = initializer
    for x in it:
        accum_value = function(accum_value, x)
    return accum_value

>>> t = ()
>>> t = t + (1,)
>>> t
(1,)
>>> t = t + (2,)
>>> t
(1, 2)
>>> t = t + (3, 4, 5)
>>> t
(1, 2, 3, 4, 5)
>>> t = t + (6, 7, 8,)
>>> t
(1, 2, 3, 4, 5, 6, 7, 8)

"""
