import tensorflow as tf
import numpy as np
import input_data

# randomly initialize nn for [encoder+bias]/[decoder+ bias]
# via training, hope to automatically adjusting weights/biases to achieve 
# reconstruction/restoration/decoding with min cost
# Tensorflow aymeric 
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden = 500
corruption_level = 0.3

# create node for input data
X = tf.placeholder("float", [None, n_visible], name='X')

# create node for corruption mask
mask = tf.placeholder("float", [None, n_visible], name='mask')
mask2 = tf.placeholder("float", [None, n_hidden], name='mask2')
# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_visible + n_hidden))
W_init = tf.random_uniform(shape=[n_visible, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.transpose(W)  # tied weights between encoder and decoder
b_prime = tf.Variable(tf.zeros([n_visible]), name='b_prime')


def model(X, mask2, W, b, W_prime, b_prime):
    # tilde_X = mask * X  # corrupted X
    # Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state
    orgY = tf.nn.sigmoid(tf.matmul(X, W) + b)  # hidden state
    Y= mask2*orgY # corrupted orgY
    Z = tf.nn.sigmoid(tf.matmul(Y, W_prime) + b_prime)  # reconstructed input
    return Z

# build model graph
Z = model(X, mask2, W, b, W_prime, b_prime)

# create cost function
cost = tf.reduce_sum(tf.pow(X - Z, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# load MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            input_ = trX[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, (input_.shape[0], n_hidden))
            sess.run(train_op, feed_dict={X: input_, mask2: mask_np})

        mask_np = np.random.binomial(1, 1 - corruption_level, (teX.shape[0], n_hidden))
        print(i, sess.run(cost, feed_dict={X: teX, mask2: mask_np}))
