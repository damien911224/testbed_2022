import tensorflow as tf


def build_model(inputs, weight_decay, end_points, dtype, dformat, is_training, scope):

    batch_norm_decay = 0.9997
    batch_norm_epsilon = 0.001

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    bias_initilizer = tf.zeros_initializer()
    bias_regularizer = None

    activation_funtion = tf.nn.relu

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        end_point = 'Conv3d_1a_1x7x7'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 7, 7,
                                            inputs.get_shape()[-1] if dformat == 'NDHWC'
                                            else inputs.get_shape()[1], 64],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, 64]
            #                          if dformat == 'NDHWC' else [1, 64, 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(inputs, kernel,
                                [1, 1, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 1, 2, 2],
                                padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            net = activation_funtion(bn)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Conv3d_1b_7x1x1'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[7, 1, 1,
                                            net.get_shape()[-1] if dformat == 'NDHWC'
                                            else net.get_shape()[1], 64],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, 64]
            #                          if dformat == 'NDHWC' else [1, 64, 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(net, kernel,
                                [1, 2, 1, 1, 1] if dformat == 'NDHWC' else [1, 1, 2, 1, 1],
                                padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            net = activation_funtion(bn)

            with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
                pooled = tf.reduce_mean(net, axis=(1, 2, 3), keepdims=True)
                kernel = tf.get_variable(name='kernel',
                                         dtype=dtype,
                                         shape=[1, 1, 1,
                                                pooled.get_shape()[-1], pooled.get_shape()[-1]],
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         trainable=is_training)
                weights = tf.nn.conv3d(pooled, kernel, [1, 1, 1, 1, 1], padding="SAME")
                biases = tf.get_variable(name='conv_3d/bias',
                                         dtype=dtype,
                                         shape=[pooled.get_shape()[-1]],
                                         initializer=bias_initilizer,
                                         regularizer=bias_regularizer,
                                         trainable=is_training)
                weights = tf.nn.bias_add(weights, biases)
                weights = tf.nn.sigmoid(weights)

                net = tf.multiply(net, weights)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'MaxPool_2a_1x3x3'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = tf.nn.max_pool3d(net,
                                   [1, 1, 3, 3, 1] if dformat == 'NDHWC' else [1, 1, 1, 3, 3],
                                   [1, 1, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 1, 2, 2],
                                   padding='SAME', data_format=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Conv3d_2b_1x1x1'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 1, 1,
                                            net.get_shape()[-1] if dformat == 'NDHWC'
                                            else net.get_shape()[1], 64],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, 64]
            #                          if dformat == 'NDHWC' else [1, 64, 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            net = activation_funtion(bn)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Conv3d_2c_1x3x3'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 3, 3,
                                            net.get_shape()[-1] if dformat == 'NDHWC'
                                            else net.get_shape()[1], 192],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, 192]
            #                          if dformat == 'NDHWC' else [1, 192, 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            net = activation_funtion(bn)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Conv3d_2d_3x1x1'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 1, 1,
                                            net.get_shape()[-1] if dformat == 'NDHWC'
                                            else net.get_shape()[1], 192],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, 192]
            #                          if dformat == 'NDHWC' else [1, 192, 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(net, kernel, [1, 1, 1, 1, 1], padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            net = activation_funtion(bn)

            with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
                pooled = tf.reduce_mean(net, axis=(1, 2, 3), keepdims=True)
                kernel = tf.get_variable(name='kernel',
                                         dtype=dtype,
                                         shape=[1, 1, 1,
                                                pooled.get_shape()[-1], pooled.get_shape()[-1]],
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         trainable=is_training)
                weights = tf.nn.conv3d(pooled, kernel, [1, 1, 1, 1, 1], padding="SAME")
                biases = tf.get_variable(name='conv_3d/bias',
                                         dtype=dtype,
                                         shape=[pooled.get_shape()[-1]],
                                         initializer=bias_initilizer,
                                         regularizer=bias_regularizer,
                                         trainable=is_training)
                weights = tf.nn.bias_add(weights, biases)
                weights = tf.nn.sigmoid(weights)

                net = tf.multiply(net, weights)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'MaxPool_3a_1x3x3'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = tf.nn.max_pool3d(net,
                                   [1, 1, 3, 3, 1] if dformat == 'NDHWC' else [1, 1, 1, 3, 3],
                                   [1, 1, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 1, 2, 2],
                                   padding='SAME', data_format=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[64, 96, 128, 16, 32, 32],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[128, 128, 192, 32, 96, 64],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'SelfAttention_3d'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = self_attention(x=net, is_training=is_training,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)

        end_point = 'MaxPool_4a_3x3x3'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = tf.nn.max_pool3d(net,
                                   [1, 3, 3, 3, 1] if dformat == 'NDHWC' else [1, 1, 3, 3, 3],
                                   [1, 2, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 2, 2, 2],
                                   padding='SAME', data_format=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[192, 96, 208, 16, 48, 64],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[160, 112, 224, 24, 64, 64],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[128, 128, 256, 24, 64, 64],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[112, 144, 288, 32, 64, 64],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[256, 160, 320, 32, 128, 128],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'SelfAttention_4g'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = self_attention(x=net, is_training=is_training,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)

        end_point = 'MaxPool_5a_2x2x2'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = tf.nn.max_pool3d(net,
                                   [1, 2, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 2, 2, 2],
                                   [1, 2, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 2, 2, 2],
                                   padding='SAME', data_format=dformat)
        end_points[end_point] = net

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[256, 160, 320, 32, 128, 128],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = inception(x=net,
                            branches=[384, 192, 384, 48, 128, 128],
                            is_training=is_training,
                            activation_funtion=activation_funtion,
                            batch_norm_decay=batch_norm_decay,
                            batch_norm_epsilon=batch_norm_epsilon,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            dtype=dtype, dformat=dformat)
        try:
            end_points[end_point] = \
                tf.concat([end_points[end_point], net], axis=0)
        except KeyError:
            end_points[end_point] = net

        end_point = 'SelfAttention_5d'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            net = self_attention(x=net, is_training=is_training,
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)

        return net


def inception(x, branches, is_training,
              activation_funtion=tf.nn.relu,
              batch_norm_decay=0.9997,
              batch_norm_epsilon=0.001,
              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
              kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-7),
              bias_initilizer=tf.zeros_initializer(),
              bias_regularizer=None,
              dtype=tf.float32, dformat='NDHWC'):
    with tf.variable_scope('Branch_0'):
        with tf.variable_scope('Conv3d_0a_1x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 1, 1,
                                            x.get_shape()[-1] if dformat == 'NDHWC'
                                            else x.get_shape()[1], branches[0]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[0]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[0], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(x, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_0 = activation_funtion(bn)

    with tf.variable_scope('Branch_1'):
        with tf.variable_scope('Conv3d_0a_1x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 1, 1,
                                            x.get_shape()[-1] if dformat == 'NDHWC'
                                            else x.get_shape()[1], branches[1]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[1]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[1], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(x, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_1 = activation_funtion(bn)

        with tf.variable_scope('Conv3d_0b_1x3x3', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 3, 3,
                                            branch_1.get_shape()[-1] if dformat == 'NDHWC'
                                            else branch_1.get_shape()[1], branches[2]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[2]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[2], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(branch_1, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_1 = activation_funtion(bn)

        with tf.variable_scope('Conv3d_0c_3x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 1, 1,
                                            branch_1.get_shape()[-1] if dformat == 'NDHWC'
                                            else branch_1.get_shape()[1], branches[2]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[2]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[2], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(branch_1, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_1 = activation_funtion(bn)

            with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
                pooled = tf.reduce_mean(branch_1, axis=(1, 2, 3), keepdims=True)
                kernel = tf.get_variable(name='kernel',
                                         dtype=dtype,
                                         shape=[1, 1, 1,
                                                pooled.get_shape()[-1], pooled.get_shape()[-1]],
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         trainable=is_training)
                weights = tf.nn.conv3d(pooled, kernel, [1, 1, 1, 1, 1], padding="SAME")
                biases = tf.get_variable(name='conv_3d/bias',
                                         dtype=dtype,
                                         shape=[pooled.get_shape()[-1]],
                                         initializer=bias_initilizer,
                                         regularizer=bias_regularizer,
                                         trainable=is_training)
                weights = tf.nn.bias_add(weights, biases)
                weights = tf.nn.sigmoid(weights)

                branch_1 = tf.multiply(branch_1, weights)

    with tf.variable_scope('Branch_2'):
        with tf.variable_scope('Conv3d_0a_1x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 1, 1,
                                            x.get_shape()[-1] if dformat == 'NDHWC'
                                            else x.get_shape()[1], branches[3]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[3]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[3], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(x, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_2 = activation_funtion(bn)

        with tf.variable_scope('Conv3d_0b_1x3x3', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 3, 3,
                                            branch_2.get_shape()[-1] if dformat == 'NDHWC'
                                            else branch_2.get_shape()[1], branches[4]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[4]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[4], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(branch_2, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_2 = activation_funtion(bn)

        with tf.variable_scope('Conv3d_0c_3x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 1, 1,
                                            branch_2.get_shape()[-1] if dformat == 'NDHWC'
                                            else branch_2.get_shape()[1], branches[4]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[4]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[4], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(branch_2, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_2 = activation_funtion(bn)

            with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
                pooled = tf.reduce_mean(branch_2, axis=(1, 2, 3), keepdims=True)
                kernel = tf.get_variable(name='kernel',
                                         dtype=dtype,
                                         shape=[1, 1, 1,
                                                pooled.get_shape()[-1], pooled.get_shape()[-1]],
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         trainable=is_training)
                weights = tf.nn.conv3d(pooled, kernel, [1, 1, 1, 1, 1], padding="SAME")
                biases = tf.get_variable(name='conv_3d/bias',
                                         dtype=dtype,
                                         shape=[pooled.get_shape()[-1]],
                                         initializer=bias_initilizer,
                                         regularizer=bias_regularizer,
                                         trainable=is_training)
                weights = tf.nn.bias_add(weights, biases)
                weights = tf.nn.sigmoid(weights)

                branch_2 = tf.multiply(branch_2, weights)

    with tf.variable_scope('Branch_3'):
        with tf.variable_scope('MaxPool_0a_3x3x3', reuse=tf.AUTO_REUSE):
            branch_3 = tf.nn.max_pool3d(x,
                                        [1, 3, 3, 3, 1] if dformat == 'NDHWC' else [1, 1, 3, 3, 3],
                                        [1, 1, 1, 1, 1],
                                        padding='SAME', data_format=dformat)

        with tf.variable_scope('Conv3d_0b_1x1x1', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[1, 1, 1,
                                            branch_3.get_shape()[-1] if dformat == 'NDHWC'
                                            else branch_3.get_shape()[1], branches[5]],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            # biases = tf.get_variable(name='conv_3d/bias',
            #                          dtype=dtype,
            #                          shape=[1, 1, 1, 1, branches[5]]
            #                          if dformat == 'NDHWC'
            #                          else [1, branches[5], 1, 1, 1],
            #                          initializer=bias_initilizer,
            #                          regularizer=bias_regularizer,
            #                          trainable=is_training)
            out = tf.nn.conv3d(branch_3, kernel, [1, 1, 1, 1, 1],
                               padding='SAME', data_format=dformat)
            # out = tf.add(out, biases)
            bn = tf.layers.batch_normalization(out,
                                               axis=-1 if dformat == 'NDHWC' else 1,
                                               center=True,
                                               scale=False,
                                               momentum=batch_norm_decay,
                                               epsilon=batch_norm_epsilon,
                                               training=is_training,
                                               trainable=is_training)
            branch_3 = activation_funtion(bn)

    net = tf.concat(axis=4 if dformat == 'NDHWC' else 1,
                    values=[branch_0, branch_1, branch_2, branch_3])

    return net


def self_attention(x, is_training,
                   num_heads=8,
                   attention_dropout_rate=0.1,
                   relu_dropout_rate=0.1,
                   post_dropout_rate=0.1,
                   relative_position=False,
                   max_relative_position=20,
                   use_bias=False,
                   use_attention_bias=True,
                   attention_bias_type="prior",
                   output_channel=None,
                   masks=None,
                   normalization_method="layer_norm",
                   activation_function=tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-7),
                   bias_initializer=tf.zeros_initializer(),
                   bias_regularizer=None,
                   dtype=tf.float32, dformat="NWC"):
    if dformat == "NWC":
        N, W, C = x.get_shape().as_list()
    else:
        N, C, W = x.get_shape().as_list()
        x = tf.transpose(x, (0, 2, 1))

    if output_channel is None:
        inner_C = C
    else:
        inner_C = output_channel

    stage_input = tf.identity(x)

    with tf.variable_scope("SelfAttention_0a", reuse=tf.AUTO_REUSE):
        outputs = tf.identity(x)

        with tf.variable_scope("Q", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="conv_1d/kernel",
                                     dtype=dtype,
                                     shape=[1, C, inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            Q = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

        with tf.variable_scope("K", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="conv_1d/kernel",
                                     dtype=dtype,
                                     shape=[1, C, inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            K = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

            if masks is not None:
                K = tf.multiply(K, tf.expand_dims(masks, axis=-1))

        with tf.variable_scope("V", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="conv_1d/kernel",
                                     dtype=dtype,
                                     shape=[1, C, inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            V = tf.nn.conv1d(outputs, kernel, [1, 1, 1], padding="SAME")

            if masks is not None:
                V = tf.multiply(V, tf.expand_dims(masks, axis=-1))

        if relative_position:
            def _generate_relative_positions_matrix(length_q, length_k,
                                                    max_relative_position):
                """Generates matrix of relative positions between inputs."""
                if length_q == length_k:
                    range_vec_q = range_vec_k = tf.range(length_q)
                else:
                    range_vec_k = tf.range(length_k)
                    range_vec_q = range_vec_k[-length_q:]
                distance_mat = range_vec_k[None, :] - range_vec_q[:, None]

                distance_mat_clipped = tf.clip_by_value(distance_mat,
                                                        -max_relative_position,
                                                        max_relative_position)
                # Shift values to be >= 0. Each integer still uniquely identifies a relative
                # position difference.
                final_mat = distance_mat_clipped + max_relative_position
                return final_mat

            def _generate_relative_positions_embeddings(length_q, length_k, depth,
                                                        max_relative_position, name):
                """Generates tensor of size [1 if cache else length_q, length_k, depth]."""
                with tf.variable_scope(name):
                    relative_positions_matrix = \
                        _generate_relative_positions_matrix(
                            length_q, length_k, max_relative_position)
                    vocab_size = max_relative_position * 2 + 1
                    # Generates embedding for each relative position of dimension depth.
                    embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
                    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
                    return embeddings

            def _relative_attention_inner(x, y, z, transpose):
                """Relative position-aware dot-product attention inner calculation.
                This batches matrix multiply calculations to avoid unnecessary broadcasting.
                Args:
                  x: Tensor with shape [batch_size, heads, length or 1, length or depth].
                  y: Tensor with shape [batch_size, heads, length or 1, depth].
                  z: Tensor with shape [length or 1, length, depth].
                  transpose: Whether to transpose inner matrices of y and z. Should be true if
                      last dimension of x is depth, not length.
                Returns:
                  A Tensor with shape [batch_size, heads, length, length or depth].
                """
                batch_size = tf.shape(x)[0]
                heads = x.get_shape().as_list()[1]
                length = tf.shape(x)[2]

                # xy_matmul is [batch_size, heads, length or 1, length or depth]
                xy_matmul = tf.matmul(x, y, transpose_b=transpose)
                # x_t is [length or 1, batch_size, heads, length or depth]
                x_t = tf.transpose(x, [2, 0, 1, 3])
                # x_t_r is [length or 1, batch_size * heads, length or depth]
                x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
                # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
                x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
                # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
                x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
                # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
                x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
                return xy_matmul + x_tz_matmul_r_t

            # split and concat
            # N x h x W x C/h
            Q = tf.concat(tf.split(tf.expand_dims(Q, axis=1), num_heads, axis=-1), axis=1)
            K = tf.concat(tf.split(tf.expand_dims(K, axis=1), num_heads, axis=-1), axis=1)
            V = tf.concat(tf.split(tf.expand_dims(V, axis=1), num_heads, axis=-1), axis=1)

            N, h, W, c = Q.get_shape().as_list()

            # Use separate embeddings suitable for keys and values
            relations_keys = \
                _generate_relative_positions_embeddings(
                    length_q=W, length_k=W, depth=c,
                    max_relative_position=max_relative_position,
                    name="relative_positions_keys")
            relations_values = \
                _generate_relative_positions_embeddings(
                    length_q=W, length_k=W, depth=c,
                    max_relative_position=max_relative_position,
                    name="relative_positions_values")

            # Compute self attention considering the relative position embeddings.
            logits = _relative_attention_inner(x=Q, y=K, z=relations_keys,
                                               transpose=True)

            if use_attention_bias:
                if attention_bias_type == "gaussian":
                    with tf.variable_scope("gaussian_bias", reuse=tf.AUTO_REUSE):
                        with tf.variable_scope("W_p", reuse=tf.AUTO_REUSE):
                            W_p = tf.get_variable(name="kernel",
                                                  dtype=dtype,
                                                  shape=[h, c, c],
                                                  initializer=kernel_initializer,
                                                  regularizer=kernel_regularizer,
                                                  trainable=is_training)

                        with tf.variable_scope("U_p", reuse=tf.AUTO_REUSE):
                            U_p = tf.get_variable(name="kernel",
                                                  dtype=dtype,
                                                  shape=[h, c],
                                                  initializer=kernel_initializer,
                                                  regularizer=kernel_regularizer,
                                                  trainable=is_training)

                        with tf.variable_scope("U_d", reuse=tf.AUTO_REUSE):
                            U_d = tf.get_variable(name="kernel",
                                                  dtype=dtype,
                                                  shape=[h, c],
                                                  initializer=kernel_initializer,
                                                  regularizer=kernel_regularizer,
                                                  trainable=is_training)

                        # N * W, h, c
                        Q_ = tf.reshape(tf.transpose(Q, (0, 2, 1, 3)), (N * W, h, c))
                        W_p_out = list()
                        for h_i in range(h):
                            out = tf.matmul(Q_[:, h_i], W_p[h_i])
                            W_p_out.append(out)
                        W_p_out = tf.nn.tanh(tf.stack(W_p_out, axis=1))
                        # N * W, h
                        P = tf.reduce_sum(tf.multiply(W_p_out, tf.expand_dims(U_p, axis=0)), axis=-1)
                        Z = tf.reduce_sum(tf.multiply(W_p_out, tf.expand_dims(U_d, axis=0)), axis=-1)
                        # N, W, h
                        P = float(W) * tf.nn.sigmoid(tf.reshape(P, (N, W, h)))
                        Z = float(W) * tf.nn.sigmoid(tf.reshape(Z, (N, W, h)))
                        # N, h, W
                        P = tf.transpose(P, (0, 2, 1))
                        Z = tf.transpose(Z, (0, 2, 1))
                        sigma = Z / 2.0
                        # N, h, W, W
                        G = tf.divide(
                            -tf.square(tf.reshape(tf.range(1, W + 1, dtype=tf.float32), (1, 1, 1, W)) -
                                       tf.expand_dims(P, axis=3)),
                            (2.0 * tf.square(tf.expand_dims(sigma, axis=3)) + 1.0e-7))
                        logits += G
                else:
                    with tf.variable_scope("prior_bias", reuse=tf.AUTO_REUSE):
                        bias = tf.get_variable(name="bias",
                                               dtype=dtype,
                                               shape=[1, 1, 1, W],
                                               initializer=bias_initializer,
                                               regularizer=bias_regularizer,
                                               trainable=is_training)

                        logits += bias

            scores = tf.nn.softmax(logits, name="attention_weights")

            if attention_dropout_rate > 0.0:
                outputs = tf.layers.dropout(scores, rate=attention_dropout_rate,
                                            training=is_training)
            else:
                outputs = tf.identity(scores)

            outputs = _relative_attention_inner(x=outputs, y=V, z=relations_values,
                                                transpose=False)

            outputs = tf.concat(tf.unstack(outputs, axis=1), axis=-1)
        else:
            # split and concat
            # hN x W x C/h
            Q = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)
            K = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)
            V = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)

            hN, W, c_h = Q.get_shape().as_list()

            # hN x C/h x W
            K = tf.transpose(K, (0, 2, 1))

            # hN x W x W
            outputs = tf.matmul(Q, K)
            outputs = tf.divide(outputs, tf.sqrt(tf.constant(c_h, dtype=dtype)))

            if use_attention_bias:
                with tf.variable_scope("attention_bias", reuse=tf.AUTO_REUSE):
                    bias = tf.get_variable(name="bias",
                                           dtype=dtype,
                                           shape=[1, 1, W],
                                           initializer=bias_initializer,
                                           regularizer=bias_regularizer,
                                           trainable=is_training)

                    outputs += bias

            outputs = tf.nn.softmax(outputs, -1)
            scores = tf.reshape(outputs, (N, num_heads, W, W))

            # hN x W x C/h
            if attention_dropout_rate > 0.0:
                outputs = tf.layers.dropout(outputs, rate=attention_dropout_rate, training=is_training)
            outputs = tf.matmul(outputs, V)

            # N x W x C
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1)

        attention_scores = scores

        with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="fc/kernel",
                                     dtype=dtype,
                                     shape=[outputs.get_shape().as_list()[-1], inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            outputs = tf.matmul(outputs, kernel)

        if post_dropout_rate > 0.0:
            outputs = tf.layers.dropout(outputs, rate=post_dropout_rate, training=is_training)

        if inner_C != C:
            with tf.variable_scope("Shortcut", reuse=tf.AUTO_REUSE):
                kernel = tf.get_variable(name="conv_1d/kernel",
                                         dtype=dtype,
                                         shape=[1, C, inner_C],
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         trainable=is_training)
                x = tf.nn.conv1d(x, kernel, [1, 1, 1], padding="SAME")
                if use_bias:
                    bias = tf.get_variable(name="conv_1d/bias",
                                           dtype=dtype,
                                           shape=[inner_C],
                                           initializer=bias_initializer,
                                           regularizer=bias_regularizer,
                                           trainable=is_training)
                    x = tf.nn.bias_add(x, bias)
                x = self.normalization(x, is_training=is_training,
                                       method=normalization_method,
                                       dformat="NDHWC" if dformat == "NWC" else "NCDHW")

        x = outputs + x

        x = self.normalization(x, is_training=is_training,
                               method=normalization_method,
                               dformat="NDHWC" if dformat == "NWC" else "NCDHW")

    with tf.variable_scope("FeedForward_0c", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("FC_0a", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="fc/kernel",
                                     dtype=dtype,
                                     shape=[inner_C, inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            outputs = tf.matmul(x, kernel)
            if use_bias:
                bias = tf.get_variable(name="fc/bias",
                                       dtype=dtype,
                                       shape=[inner_C],
                                       initializer=bias_initializer,
                                       regularizer=bias_regularizer,
                                       trainable=is_training)
                outputs = tf.nn.bias_add(outputs, bias)
            outputs = self.normalization(outputs, is_training=is_training,
                                         method=normalization_method,
                                         dformat="NDHWC" if dformat == "NWC" else "NCDHW")
            outputs = activation_function(outputs)

            if relu_dropout_rate > 0.0:
                outputs = tf.layers.dropout(outputs, rate=relu_dropout_rate, training=is_training)

        with tf.variable_scope("FC_0b", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="fc/kernel",
                                     dtype=dtype,
                                     shape=[outputs.get_shape().as_list()[-1], inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            outputs = tf.matmul(outputs, kernel)
            if use_bias:
                bias = tf.get_variable(name="fc/bias",
                                       dtype=dtype,
                                       shape=[inner_C],
                                       initializer=bias_initializer,
                                       regularizer=bias_regularizer,
                                       trainable=is_training)
                outputs = tf.nn.bias_add(outputs, bias)

        if post_dropout_rate > 0.0:
            outputs = tf.layers.dropout(outputs, rate=post_dropout_rate, training=is_training)

        outputs += x

        outputs = self.normalization(outputs, is_training=is_training,
                                     method=normalization_method,
                                     dformat="NDHWC" if dformat == "NWC" else "NCDHW")

    if inner_C != C:
        with tf.variable_scope("Shortcut", reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name="conv_1d/kernel",
                                     dtype=dtype,
                                     shape=[1, C, inner_C],
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer,
                                     trainable=is_training)
            stage_input = tf.nn.conv1d(stage_input, kernel, [1, 1, 1], padding="SAME")
            if use_bias:
                bias = tf.get_variable(name="conv_1d/bias",
                                       dtype=dtype,
                                       shape=[inner_C],
                                       initializer=bias_initializer,
                                       regularizer=bias_regularizer,
                                       trainable=is_training)
                stage_input = tf.nn.bias_add(stage_input, bias)
            stage_input = \
                self.normalization(stage_input, is_training=is_training,
                                   method=normalization_method,
                                   dformat="NDHWC" if dformat == "NWC" else "NCDHW")

    outputs += stage_input

    outputs = self.normalization(outputs, is_training=is_training,
                                 method=normalization_method,
                                 dformat="NDHWC" if dformat == "NWC" else "NCDHW")

    if dformat == "NCW":
        outputs = tf.transpose(outputs, (0, 2, 1))

    return outputs
