# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt

shape = (10, 10)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

with tf.Session() as session:
    X = session.run(initial_board)
    print(X[0][0:10], X[1][0:10])

# fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
# cmap colour scheme. In this case, using ‘Greys’ results in a black and white matrix, 
# plt.show()
plt.savefig("lesson8-board-ini.png")

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
"""
 the results from running board_update are a list of matrices, 
 even though our function defined only return a single value.
"""

with tf.Session() as session:
    initial_board_values = session.run(initial_board)
    X = session.run(board_update, feed_dict={board: initial_board_values})[0]
    
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')
# cmap colour scheme. In this case, using ‘Greys’ results in a black and white matrix, 
# plt.show()
plt.savefig("lesson8-board-update1.png")
    