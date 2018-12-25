from models.layers import weight_variable,bias_variable,\
    conv2d_activtion,max_pool,deconv2d_concat,conv2d
import tensorflow as tf

def unet(x, batch_norm=True, n_class=5, features=64):
    '卷积之后->norm->elu'
    x = x/255

    with tf.name_scope('1.unit'):
        w1_1 = weight_variable([3, 3, 3, features])
        b1_1 = bias_variable([features])
        net = conv2d_activtion(x, w1_1, b1_1, batch_norm)
        w1_2 = weight_variable([3, 3, features, features])
        b1_2 = bias_variable([features])
        net = conv2d_activtion(net, w1_2, b1_2, batch_norm)
        crop_and_concat1 = net
        net = max_pool(net)

    features = features*2
    with tf.name_scope('2.unit'):
        w2_1 = weight_variable([3, 3, features//2, features])
        b2_1 = bias_variable([features])
        net = conv2d_activtion(net, w2_1, b2_1, batch_norm)
        w2_2 = weight_variable([3, 3, features, features])
        b2_2 = bias_variable([features])
        net = conv2d_activtion(net, w2_2, b2_2, batch_norm)
        crop_and_concat2 = net
        net = max_pool(net)

    features = features*2
    with tf.name_scope('3.unit'):
        w3_1 = weight_variable([3, 3, features//2, features])
        b3_1 = bias_variable([features])
        net = conv2d_activtion(net, w3_1, b3_1, batch_norm)
        w3_2 = weight_variable([3, 3, features, features])
        b3_2 = bias_variable([features])
        net = conv2d_activtion(net, w3_2, b3_2, batch_norm)
        crop_and_concat3 = net
        net = max_pool(net)

    features = features*2
    with tf.name_scope('4.unit'):
        w4_1 = weight_variable([3, 3, features//2, features])
        b4_1 = bias_variable([features])
        net = conv2d_activtion(net, w4_1, b4_1, batch_norm)
        w4_2 = weight_variable([3, 3, features, features])
        b4_2 = bias_variable([features])
        net = conv2d_activtion(net, w4_2, b4_2, batch_norm)
        crop_and_concat4 = net
        net = max_pool(net)

    features = features*2
    with tf.name_scope('5.unit'):
        w5_1 = weight_variable([3, 3, features//2, features])
        b5_1 = bias_variable([features])
        net = conv2d_activtion(net, w5_1, b5_1, batch_norm)
        w5_2 = weight_variable([3, 3, features, features])
        b5_2 = bias_variable([features])
        net = conv2d_activtion(net, w5_2, b5_2, batch_norm)

    'up'
    features = features//2
    with tf.name_scope('6.unit'):
        wd4_1 = weight_variable([2, 2, features, features*2])
        bd4_1 = bias_variable([features])
        net = deconv2d_concat(net, wd4_1, bd4_1, crop_and_concat4)
        wd4_2 = weight_variable([3, 3, features*2, features])
        bd4_2 = bias_variable([features])
        net = conv2d_activtion(net, wd4_2, bd4_2, batch_norm)
        wd4_3 = weight_variable([3, 3, features, features])
        bd4_3 = bias_variable([features])
        net = conv2d_activtion(net, wd4_3, bd4_3, batch_norm)

    features = features//2
    with tf.name_scope('7.unit'):
        wd3_1 = weight_variable([2, 2, features, features*2])
        bd3_1 = bias_variable([features])
        net = deconv2d_concat(net, wd3_1, bd3_1, crop_and_concat3)
        wd3_2 = weight_variable([3, 3, features*2, features])
        bd3_2 = bias_variable([features])
        net = conv2d_activtion(net, wd3_2, bd3_2, batch_norm)
        wd3_3 = weight_variable([3, 3, features, features])
        bd3_3 = bias_variable([features])
        net = conv2d_activtion(net, wd3_3, bd3_3, batch_norm)

    features = features//2
    with tf.name_scope('8.unit'):
        wd2_1 = weight_variable([2, 2, features, features*2])
        bd2_1 = bias_variable([features])
        net = deconv2d_concat(net, wd2_1, bd2_1, crop_and_concat2)
        wd2_2 = weight_variable([3, 3, features*2, features])
        bd2_2 = bias_variable([features])
        net = conv2d_activtion(net, wd2_2, bd2_2, batch_norm)
        wd2_3 = weight_variable([3, 3, features, features])
        bd2_3 = bias_variable([features])
        net = conv2d_activtion(net, wd2_3, bd2_3, batch_norm)

    features = features//2
    with tf.name_scope('9.unit'):
        wd1_1 = weight_variable([2, 2, features, features*2])
        bd1_1 = bias_variable([features])
        net = deconv2d_concat(net, wd1_1, bd1_1, crop_and_concat1)
        wd1_2 = weight_variable([3, 3, features*2, features])
        bd1_2 = bias_variable([features])
        net = conv2d_activtion(net, wd1_2, bd1_2, batch_norm)
        wd1_3 = weight_variable([3, 3, features, features])
        bd1_3 = bias_variable([features])
        net = conv2d_activtion(net, wd1_3, bd1_3, batch_norm)

    with tf.name_scope('10.unit'):
        w_out = weight_variable([1, 1, features, n_class])
        b_out = bias_variable([n_class])
        output_map = conv2d(net, w_out, b_out)
    return output_map