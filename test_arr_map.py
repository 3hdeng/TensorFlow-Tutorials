# http://stackoverflow.com/questions/6824122/mapping-a-numpy-array-in-place
import timeit
from numpy import array, arange, vectorize, rint, multiply, round as np_round 

# SETUP
get_array = lambda side : arange(side**2).reshape(side, side) * 30
dim = lambda x : int(round(x * 0.67328))

# TIMER
def best(fname, reps, side):
    global a
    a = get_array(side)
        t = timeit.Timer('%s(a)' % fname,
                     setup='from __main__ import %s, a' % fname)
    return min(t.repeat(reps, 3))  #low num as in place --> converge to 1

# FUNCTIONS
def mac(array_):
    for row in range(len(array_)):
        for col in range(len(array_[0])):
            array_[row][col] = dim(array_[row][col])

def mac_two(array_):
    li = range(len(array_[0]))
    for row in range(len(array_)):
        for col in li:
            array_[row][col] = int(round(array_[row][col] * 0.67328))

def mac_three(array_):
    for i, row in enumerate(array_):
        array_[i][:] = [int(round(v * 0.67328)) for v in row]


def senderle(array_):
    array_ = array_.reshape(-1)
    for i, v in enumerate(array_):
        array_[i] = dim(v)

def eryksun(array_):
    array_[:] = vectorize(dim)(array_)

def ufunc_ed(array_):
    multiplied = array_ * 0.67328
    array_[:] = rint(multiplied)

# fmilo has problem
def fmilo(array_):
    np_round(multiply(array_ ,0.67328, array_), out=array_)
    
# MAIN
r = []
for fname in ('mac', 'mac_two', 'mac_three', 'senderle', 'eryksun', 'ufunc_ed'):
    print('\nTesting `%s`...' % fname)
    r.append(best(fname, reps=50, side=50))
    # The following is for visually checking the functions returns same results
    tmp = get_array(3)
    eval('%s(tmp)' % fname)
    print tmp
tmp = min(r)/100
print('\n===== ...AND THE WINNER IS... =========================')
print('  mac (as in question)       :  %.4fms [%.0f%%]') % (r[0]*1000,r[0]/tmp)
print('  mac (optimised)            :  %.4fms [%.0f%%]') % (r[1]*1000,r[1]/tmp)
print('  mac (slice-assignment)     :  %.4fms [%.0f%%]') % (r[2]*1000,r[2]/tmp)
print('  senderle                   :  %.4fms [%.0f%%]') % (r[3]*1000,r[3]/tmp)
print('  eryksun                    :  %.4fms [%.0f%%]') % (r[4]*1000,r[4]/tmp)
print('  slice-assignment w/ ufunc  :  %.4fms [%.0f%%]') % (r[5]*1000,r[5]/tmp)
print('=======================================================\n')


"""
The output of the above script - at least in my system - is:

  mac (as in question)       :  88.7411ms [74591%]
  mac (optimised)            :  86.4639ms [72677%]
  mac (slice-assignment)     :  79.8671ms [67132%]
  senderle                   :  85.4590ms [71832%]
  eryksun                    :  13.8662ms [11655%]
  slice-assignment w/ ufunc  :  0.1190ms [100%]
As you can observe, using numpy's ufunc increases speed of more than 2 and almost 3 orders of magnitude compared with the second best and worst alternatives respectively.

If using ufunc is not an option, here's a comparison of the other alternatives only:

  mac (as in question)       :  91.5761ms [672%]
  mac (optimised)            :  88.9449ms [653%]
  mac (slice-assignment)     :  80.1032ms [588%]
  senderle                   :  86.3919ms [634%]
  eryksun                    :  13.6259ms [100%]


fmilo :
When you pass an output array to a numpy ufunc via out, 
it automatically casts the result to the type of the output array. 

So in this case, the floating point result is cast to an int (and thus truncated) 
before being stored. So fmilo is functionally equivalent to array_ *= 0.67328. 
To get the desired rounding behavior, you have to do something like rint((array_ * 0.67328), array_)

"""  