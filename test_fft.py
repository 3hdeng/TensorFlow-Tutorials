import numpy as np
import mytime

"""
x=np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
print x
y=np.fft.fft([1,2,3,4],4)
print y

convert a 2D numpy array to a 2D numpy matrix
If a is your array, np.asmatrix(a) is a matrix.

The fastest way is to do a*a or a**2 or np.square(a) whereas np.power(a, 2) showed to be considerably slower.

np.power() allows you to use different exponents for each element if instead of 2 you pass another 
that NumPy doesn't easily or efficiently do this the way a python list will.

import numpy as np
a = np.array([[1,3,4],[1,2,3],[1,2,1]])
b = np.array([10,20,30])
c = np.hstack((a, np.atleast_2d(b).T))
returns c:

array([[ 1,  3,  4, 10],
       [ 1,  2,  3, 20],
       [ 1,  2,  1, 30]])

Appending a row could be done by:

c = np.vstack ((a, [x] * len (a[0]) ))
returns c as:

array([[ 1,  3,  4],
       [ 1,  2,  3],
       [ 1,  2,  1],
       [10, 10, 10]])       
       
"""
N=16
x=np.arange(N)
print x
w0=np.ones(N)
print w0


w=np.exp(2j * np.pi * np.arange(N) / N)
W= np.array([w0, w])
for i in range(2,16):
    print i
    w = w*w
    W=np.vstack((W, w))
    
print W.shape

W=np.asmatrix(W/np.sqrt(N))
#t=np.dot(W,W.transpose())
t=np.dot(W,W.getH())
print np.rint(t)


    

for i in range(100000):
 x=np.random.uniform(-1,1, 16)
 #y= np.fft.fft(x)
 y=x
 if (i % 10000) ==0 :
     print x
     print y
 



"""
>>> numpy.random.uniform(-1, 1, size=10)
array([-0.92592953, -0.6045348 , -0.52860837,  0.00321798,  0.16050848,
       -0.50421058,  0.06754615,  0.46329675, -0.40952318,  0.49804386])
       
       
>>> np.random.rand(5)
array([ 0.69093485,  0.24590705,  0.02013208,  0.06921124,  0.73329277])
It also allows to generate samples in a given shape:

>>> np.random.rand(3,2)
array([[ 0.14022471,  0.96360618], 
       [ 0.37601032,  0.25528411], 
       [ 0.49313049,  0.94909878]])
As You said, uniformly distributed random numbers between [-1, 1) can be generated with:

>>> 2 * np.random.rand(5) - 1
array([ 0.86704088, -0.65406928, -0.02814943,  0.74080741, -0.14416581])
"""

