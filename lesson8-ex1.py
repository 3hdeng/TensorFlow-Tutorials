# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt

shape = (10, 10)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

#==================================
import numpy as np
from scipy.signal import convolve2d

def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X
    
board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])


import matplotlib.animation as animation

#x_value=[]
session=tf.Session()
#with tf.Session() as session:
x_value = session.run(initial_board)
    
print(x_value[0][0:10], x_value[1][0:10])
fig = plt.figure()
imgplot = plt.imshow(x_value, cmap='Greys',  interpolation='nearest')
    
def game_of_life(*args):
    X = session.run(board_update, feed_dict={board: x_value})[0]
    # ??? UnboundLocalError: local variable 'X' referenced before assignment
    # recursive fucntion call lead to UnboundLocalError: local variable 'X' referenced before assignment
    imgplot.set_array(X)
    return imgplot,

ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
# plt.show()
# Hint: you will need to remove the plt.show() from the earlier code to make this run!
session.close()