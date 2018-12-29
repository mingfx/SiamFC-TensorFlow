"""
2017/11/24 ref:https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py
"""

import tensorflow as tf
from tensorflow.python.training import moving_averages

slim = tf.contrib.slim

UPDATE_OPS_COLLECTION = "_update_ops_"

# create variable
def create_variable(name, shape, initializer,
    dtype=tf.float32, trainable=True):
    return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=initializer, trainable=trainable)

# batchnorm layer
def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
    inputs_shape = inputs.get_shape().as_list()
    params_shape = inputs_shape[-1:]
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_variable("moving_mean", params_shape,
                            initializer=tf.zeros_initializer(), trainable=False)
        moving_variance = create_variable("moving_variance", params_shape,
                            initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)

# depthwise conv2d layer
def depthwise_conv2d(inputs, scope, filter_size=3, channel_multiplier=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, channel_multiplier],
                       initializer=tf.truncated_normal_initializer(stddev=0.01))

    return tf.nn.depthwise_conv2d(inputs, filter, strides=[1, strides, strides, 1],
                                padding="SAME", rate=[1, 1])

# conv2d layer
def conv2d(inputs, scope, num_filters, filter_size=1, strides=1):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    with tf.variable_scope(scope):
        filter = create_variable("filter", shape=[filter_size, filter_size,
                                                  in_channels, num_filters],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    return tf.nn.conv2d(inputs, filter, strides=[1, strides, strides, 1],
                        padding="SAME")

# avg pool layer
def avg_pool(inputs, pool_size, scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1],
                strides=[1, pool_size, pool_size, 1], padding="VALID")

# fully connected layer
def fc(inputs, n_out, scope, use_bias=True):
    inputs_shape = inputs.get_shape().as_list()
    n_in = inputs_shape[-1]
    with tf.variable_scope(scope):
        weight = create_variable("weight", shape=[n_in, n_out],
                    initializer=tf.random_normal_initializer(stddev=0.01))
        if use_bias:
            bias = create_variable("bias", shape=[n_out,],
                                   initializer=tf.zeros_initializer())
            return tf.nn.xw_plus_b(inputs, weight, bias)
        return tf.matmul(inputs, weight)


def mobileNet(inputs, num_classes=1000, is_training=True,
              width_multiplier=1, scope="MobileNet", reuse=None):
    """
    The implement of MobileNet(ref:https://arxiv.org/abs/1704.04861)
    :param inputs: 4-D Tensor of [batch_size, height, width, channels]
    :param num_classes: number of classes
    :param is_training: Boolean, whether or not the model is training
    :param width_multiplier: float, controls the size of model
    :param scope: Optional scope for variables
        """
    num_classes = num_classes
    is_training = is_training
    width_multiplier = width_multiplier

    # construct model
    with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        # conv1
        with slim.arg_scope([_depthwise_separable_conv2d, avg_pool],
                            outputs_collections=end_points_collection):
            net = conv2d(inputs, "conv_1", round(32 * width_multiplier), filter_size=3,
                         strides=2)  # ->[N, 112, 112, 32]
            net = tf.nn.relu(bacthnorm(net, "conv_1/bn", is_training=is_training))
            net = _depthwise_separable_conv2d(net, 64, width_multiplier,
                                    "ds_conv_2") # ->[N, 112, 112, 64]
            net = _depthwise_separable_conv2d(net, 128, width_multiplier,
                                    "ds_conv_3", downsample=True) # ->[N, 56, 56, 128]
            net = _depthwise_separable_conv2d(net, 128, width_multiplier,
                                    "ds_conv_4") # ->[N, 56, 56, 128]
            net = _depthwise_separable_conv2d(net, 256, width_multiplier,
                                    "ds_conv_5", downsample=True) # ->[N, 28, 28, 256]
            net = _depthwise_separable_conv2d(net, 256, width_multiplier,
                                    "ds_conv_6") # ->[N, 28, 28, 256]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_7", downsample=True) # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_8") # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_9")  # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_10")  # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_11")  # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 512, width_multiplier,
                                    "ds_conv_12")  # ->[N, 14, 14, 512]
            net = _depthwise_separable_conv2d(net, 1024, width_multiplier,
                                    "ds_conv_13", downsample=True) # ->[N, 7, 7, 1024]
            net = _depthwise_separable_conv2d(net, 1024, width_multiplier,
                                    "ds_conv_14") # ->[N, 7, 7, 1024]
            net = avg_pool(net, 7, "avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            logits = fc(net, num_classes, "fc_16")
            predictions = tf.nn.softmax(logits)
            return net

def _depthwise_separable_conv2d(inputs, num_filters, width_multiplier,
                                    scope, downsample=False):
    """depthwise separable convolution 2D function"""
    num_filters = round(num_filters * width_multiplier)
    strides = 2 if downsample else 1

    with tf.variable_scope(scope):
        # depthwise conv2d
        dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
        #为什么要使用Batch Norm呢？上边所说的加快训练速度只是一个简单的原因，在简单的深层网络中，如果前层中的参数改变，后层中的参数也会跟着变化，如果加上Batch Norm，即使输入数据的分布会有变化，但是他们的均值方差可控，从而使变化带来的影响减小，使各个层之间更加独立，更利于每层‘专门做自己的事情’。
        # batchnorm
        bn = bacthnorm(dw_conv, "dw_bn", is_training=is_training)
        # relu
        relu = tf.nn.relu(bn)
        # pointwise conv2d (1x1)
        pw_conv = conv2d(relu, "pointwise_conv", num_filters)
        # bn
        bn = bacthnorm(pw_conv, "pw_bn", is_training=is_training)
        return tf.nn.relu(bn)
