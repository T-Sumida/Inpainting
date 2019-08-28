# coding:utf-8

import tensorflow as tf
from inpaint_ops import double_conv, convSN2d, reduce_std, softmax


def encoder(input_l, name, bn=False):
    with tf.variable_scope(name):
        X = tf.layers.conv2d(inputs=input_l, filters=32,
                             kernel_size=[5, 5], strides=(1, 1), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=64,
                             kernel_size=[3, 3], strides=(2, 2), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=64,
                             kernel_size=[3, 3], strides=(1, 1), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=128,
                             kernel_size=[3, 3], strides=(2, 2), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=128,
                             kernel_size=[3, 3], strides=(1, 1), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=256, kernel_size=[
                             3, 3], strides=(2, 2), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=256,
                             kernel_size=[3, 3], strides=(1, 1),
                             dilation_rate=(2, 2), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=256,
                             kernel_size=[3, 3], strides=(1, 1),
                             dilation_rate=(4, 4), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=256,
                             kernel_size=[3, 3], strides=(1, 1),
                             dilation_rate=(8, 8), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.conv2d(inputs=X, filters=256,
                             kernel_size=[3, 3], strides=(1, 1),
                             dilation_rate=(16, 16), padding='SAME')
        X = tf.nn.relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        return X


def decoder(input_l, name, width, height, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        X = double_conv(input_l, [128, 128], (1, 1), [3, 3])
        X = tf.image.resize_nearest_neighbor(X, (height//4, width//4))

        X = double_conv(input_l, [64, 64], (1, 1), [3, 3])
        X = tf.image.resize_nearest_neighbor(X, (height//2, width//2))

        X = double_conv(input_l, [32, 32], (1, 1), [3, 3])
        X = tf.image.resize_nearest_neighbor(X, (height, width))

        X = double_conv(input_l, [16, 16], (1, 1), [3, 3])
        X = tf.image.resize_nearest_neighbor(X, (height, width))

        X = tf.layers.conv2d(X, 3, [3, 3],
                             strides=(1, 1), padding='SAME')
        X = tf.clip_by_value(X, -1, 1)
        return X


def discriminator(input_l, name, bn=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        X = convSN2d(input_l, 64, 5, 2, 'red_convSN2d_1')
        X = tf.nn.leaky_relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = convSN2d(input_l, 128, 5, 2, 'red_convSN2d_2')
        X = tf.nn.leaky_relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = convSN2d(input_l, 256, 5, 2, 'red_convSN2d_3')
        X = tf.nn.leaky_relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = convSN2d(input_l, 256, 5, 2, 'red_convSN2d_4')
        X = tf.nn.leaky_relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = convSN2d(input_l, 512, 5, 2, 'red_convSN2d_5')
        X = tf.nn.leaky_relu(X)
        if bn:
            X = tf.layers.BatchNormalization()(X)

        X = tf.layers.flatten(X)
        X = tf.layers.dense(X, 1)
        return X


def contextual_attention(bg_in, fg_in, mask, k_size, lamda, name, stride=1):
    with tf.variable_scope(name):
        b, h, w, dims = [i.value for i in bg_in.get_shape()]
        temp = tf.image.resize_nearest_neighbor(mask, (h, w))
        temp = tf.expand_dims(temp[:, :, :, 0], 3)
        mask_r = tf.tile(temp, [1, 1, 1, dims])  # b 128 128 128
        bg = bg_in * mask_r

        kn = int((k_size - 1) / 2)
        c = 0
        for p in range(kn, h - kn, stride):
            for q in range(kn, w - kn, stride):
                c += 1

        patch1 = tf.extract_image_patches(bg, [1, k_size, k_size, 1], [
                                          1, stride, stride, 1], [1, 1, 1, 1], 'VALID')

        patch1 = tf.reshape(patch1, (b, 1, c, k_size*k_size*dims))
        patch1 = tf.reshape(patch1, (b, 1, 1, c, k_size * k_size * dims))
        patch1 = tf.transpose(patch1, [0, 1, 2, 4, 3])

        patch2 = tf.extract_image_patches(fg_in, [1, k_size, k_size, 1], [
                                          1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
        Xs = []

        for ib in range(b):

            k1 = patch1[ib, :, :, :, :]
            k1d = tf.reduce_sum(tf.square(k1), axis=2)
            k2 = tf.reshape(k1, (k_size, k_size, dims, c))
            ww = patch2[ib, :, :, :]
            wwd = tf.reduce_sum(tf.square(ww), axis=2, keepdims=True)
            ft = tf.expand_dims(ww, 0)

            x = tf.nn.conv2d(ft, k1, strides=[1, 1, 1, 1], padding='SAME')

            tt = k1d + wwd

            x = tf.expand_dims(tt, 0) - 2 * x

            x = (x - tf.reduce_mean(x, 3, True)) / \
                reduce_std(x, 3, True)
            x = -1 * tf.nn.tanh(x)

            x = softmax(lamda * x)

            Xs_t = tf.nn.conv2d_transpose(x, k2, output_shape=[1, h, w, dims],
                                          strides=[1, 1, 1, 1], padding='SAME')
            Xs_t = Xs_t / (k_size ** 2)

            if ib == 0:
                Xs = Xs_t
            else:
                Xs = tf.concat((Xs, Xs_t), 0)

        Xs = bg + Xs * (1.0 - mask_r)

        con1 = tf.concat([bg_in, Xs], 3)
        X = tf.layers.conv2d(con1, dims, [1, 1], strides=(1, 1),
                             padding='VALID')
        X = tf.nn.relu(X)

        return X


# batch_size = 64
# Width = 224
# Height = 224


# X = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])
# Y = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])

# mask = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])

# en_input = tf.concat([X, mask], 3)

# x = encoder(en_input, 'encoder')
# x_con = contextual_attention(x, x, mask, 3, 50, 'contextual_attention')

# print("x:{}".format(x.get_shape()))
# print("x_con:{}".format(x_con.get_shape()))
# decoder_i = decoder(x_con, 'decoder', Width, Height, reuse=False)
# decoder_c = decoder(x, 'decoder', Width, Height, reuse=True)

# print("decoder_c:{}".format(decoder_c.get_shape()))
# print("decoder_i:{}".format(decoder_i.get_shape()))
# discriminator_red = discriminator(decoder_i, 'red')
