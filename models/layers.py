import tensorflow as tf


def conv2d_activtion(x, w, b, batchnorm=True):
    net = conv2d(x, w, b)

    if batchnorm == True:
        net = norm(net)
    return tf.nn.elu(net)


def deconv2d_concat(x, w, b, concat):
    net = deconv2d(x, w) + b
    net = tf.concat([concat, net], 3)
    return net


def norm(x):
    return tf.layers.batch_normalization(x)


def weight_variable(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W, b, padding='SAME'):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
    return conv_2d + b



def deconv2d(x, W, stride=2):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')


def max_pool(x, n=2):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')


def crop_and_concat(x1, x2):
    # x1_shape = tf.shape(x1)
    # x2_shape = tf.shape(x2)
    # # offsets for the top left corner of the crop
    # offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    # size = [-1, x2_shape[1], x2_shape[2], -1]
    # x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1, x2], 3)


