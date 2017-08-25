''' TensorFlow implementation of denoising autoencoder (dA) 

This code is TensorFlow implementation of denoising autoencoder with an architecture described in
Convolution by Evolution (Fernando, et al.) from Google DeepMind. The purpose of this code is to
evaluate the performance of DPPN-encoded version of dA with the same architecture.

'''

import argparse
import numpy as np
import tensorflow as tf

from tensorflow.contrib.keras.python.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

# add channels to input data and normalize to [0, 1]
x_train = x_train[:, :, :, np.newaxis] / 255.0
x_test = x_test[:, :, :, np.newaxis] / 255.0


def weight_variable(shape):
    init_val = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_val, name='W')

def bias_variable(shape):
    init_val = tf.constant(0.1, shape=shape)
    return tf.Variable(init_val, name='bias')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='VALID')

def conv2d_transpose(x, w, output_shape):
    return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, 2, 2, 1], padding='VALID')


with tf.name_scope('denoising_autoencoder'):
    data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='data')
    noisy_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='noisy_data')
    tf.summary.image('noisy_data', noisy_data)

    # encoding layer
    with tf.name_scope('encoder'):
        w_enc = weight_variable([7, 7, 1, 2])
        b_enc = bias_variable([2])
        encoded = tf.nn.relu(conv2d(noisy_data, w_enc) + b_enc) 

    # decoding layer
    with tf.name_scope('decoder'):
        w_dec = weight_variable([7, 7, 1, 2])
        b_enc = bias_variable([1])
        decoded_shape = tf.stack([tf.shape(encoded)[0], 28, 28, 1])
        decoded = tf.nn.relu(conv2d_transpose(encoded, w_dec, decoded_shape) + b_enc)
    tf.summary.image('decoded', decoded)      
 
    # loss function
    with tf.name_scope('mse'):
        mse = tf.reduce_mean(tf.losses.mean_squared_error(data, decoded))
    tf.summary.scalar('mse', mse)

    # optimizer
    with tf.name_scope('train_op'):
        train_step = tf.train.AdamOptimizer().minimize(mse)

    merged = tf.summary.merge_all()

n_iter = 6000
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', type=int, default=6000, help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
parser.add_argument('--logdir', type=str, default='dA_log', help='TensorFlow log directory')

args = parser.parse_args()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./%s' % args.logdir, graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(args.n_iter):
        batch = np.random.randint(60000, size=args.batch_size)

        data_batch = x_train[batch]
        noisy_data_batch = data_batch + np.random.random(size=data_batch.shape)

        feed_dict = {noisy_data: noisy_data_batch, data: data_batch}
        summary_str, loss_val, _ = sess.run([merged, mse, train_step], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i) 

        if i % (n_iter / 10) == 0:
            print('iter %d: loss = %f' % (i, loss_val))

    noisy_data_test = x_test + np.random.random(size=x_test.shape)
    test_loss = sess.run(mse, feed_dict={noisy_data: noisy_data_test, data: x_test})
    print('test loss = %f' % test_loss)


