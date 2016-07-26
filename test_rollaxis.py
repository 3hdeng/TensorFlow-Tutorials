# test np.rollaxis()
# http://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing
# Roll a.shape[axis] to the position before a.shape[start]
# before in this context means the same as in list insert(). 
# So it is possible to insert before the end.

import numpy as np
a=np.ones((1,2,3,4,5,6))

print( np.rollaxis(a,-2).shape )
print( np.rollaxis(a,-2, 0).shape )
print( np.rollaxis(a,-2, 1).shape )
print( np.rollaxis(a,-2, 6).shape )
print( np.rollaxis(a,-2,-1).shape )
print("a.shape not changed")
print(a.shape)


"""
a = np.arange(1*2*3*4*5).reshape(1,2,3,4,5)
np.rollaxis(a,axis,start)


The basic action of rollaxis is:

axes = list(range(0, n))
axes.remove(axis)
axes.insert(start, axis)
return a.transpose(axes)
If axis<start, then start-=1 to account for the remove action.If

"""