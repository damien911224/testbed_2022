import tensorflow as tf
import os


def build_model(inputs, weight_decay, end_points, dtype, dformat, is_training, scope):

    batch_norm_decay = 0.9997
    batch_norm_epsilon = 0.001

    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    # bias_initilizer = tf.zeros_initializer()
    # bias_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    activation_funtion = tf.nn.relu

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        end_point = 'Conv3d_1a_7x7x7'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[7, 7, 7,
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
                                [1, 2, 2, 2, 1] if dformat == 'NDHWC' else [1, 1, 2, 2, 2],
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

        end_point = 'Conv3d_2c_3x3x3'
        with tf.variable_scope(end_point, reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 3, 3,
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

        return net, end_points


def inception(x, branches, is_training,
              activation_funtion=tf.nn.relu,
              batch_norm_decay=0.9997,
              batch_norm_epsilon=0.001,
              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
              kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0e-7),
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

        with tf.variable_scope('Conv3d_0b_3x3x3', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 3, 3,
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

        with tf.variable_scope('Conv3d_0b_3x3x3', reuse=tf.AUTO_REUSE):
            kernel = tf.get_variable(name='conv_3d/kernel',
                                     dtype=dtype,
                                     shape=[3, 3, 3,
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


if __name__ == '__main__':

    frames = \
        tf.placeholder(dtype=tf.float32, shape=(8,
                                                3,
                                                16,
                                                224,
                                                224))

    end_points = build_model(frames, 1.0e-7, tf.float32, 'NCDHW', True, 'I3D_RGB')

    # load_ckpt_file_path = os.path.join('/home/damien/testbed_20171107',
    #                                     'cnn/I3D/rgb', 'model.ckpt')
    #
    # loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='I3D_RGB'))
    #
    # with tf.Session() as session:
    #     print 'Loading Pre-trained Models ...'
    #     loader.restore(session, load_ckpt_file_path)
    #     print 'Pre-trained Models are Loaded!'
    #
    #     print(session.run(end_points['Mixed_5c'],
    #                       feed_dict={frames: np.zeros(dtype=np.float32,
    #                                                   shape=(8, 3, 16, 224, 224))}))
    #
    #     print('')

    load_ckpt_file_path = os.path.join('/home/damien/testbed_20171107',
                                       'cnn/I3D/pretrained/rgb_imagenet', 'model.ckpt')

    load_parameters = dict()
    for param in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='I3D_RGB'):
        replaced_name = \
            param.name.replace('I3D_RGB', 'RGB/inception_i3d').replace(':0', '').replace('kernel', 'w').replace('7x7x7', '7x7').replace('1x1x1', '1x1').replace('3x3x3', '3x3')
        replaced_name = replaced_name.replace('1x3x3', '3x3').replace('2x2x2', '2x2').replace('batch_normalization', 'batch_norm')
        replaced_name = replaced_name.replace('Mixed_5b/Branch_2/Conv3d_0b_3x3', 'Mixed_5b/Branch_2/Conv3d_0a_3x3')

        if 'gamma' not in replaced_name and 'bias' not in replaced_name and 'Up' not in replaced_name:
            load_parameters[replaced_name] = param

    loader = tf.train.Saver(var_list=load_parameters)
    saver = tf.train.Saver(var_list=load_parameters.values())

    with tf.Session() as session:
        # init_variables = tf.global_variables_initializer()
        # session.run(init_variables)

        print('Loading Pre-trained Models ...')
        loader.restore(session, load_ckpt_file_path)
        print('Pre-trained Models are Loaded!')

        saver.save(session, '/home/damien/testbed_20171107/cnn/I3D/rgb/model.ckpt')
