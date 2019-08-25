# coding:utf-8
import tensorflow as tf
import numpy as np
import os
import scipy.io
import urllib.request
import urllib.parse


def load_image(self, mat_path, load_image_num, save_path="./data"):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    mat_data = scipy.io.loadmat(mat_path)
    sun_urls = mat_data['SUN']
    cntr = 0

    divide = load_image_num // 900
    kind_counter = 0
    for urls in sun_urls:
        for u in urls:
            print(kind_counter)
            kind_counter += 1
            cntr = 0
            for p in u[2][0]:
                try:
                    img_data = urllib.request.urlopen(p[0]).read()
                    file_name = "{:05}_{:010}.jpg".format(
                        kind_counter, cntr)
                    f = open(os.path.join(
                        save_path, file_name), 'wb')

                    f.write(img_data)
                    f.close()
                    cntr += 1
                    if cntr == divide:
                        break

                except Exception as e:
                    print(e)


def double_conv(input_l, filter_sizes, strides, kernel_sizes):
    X = tf.layers.conv2d(inputs=input_l, filters=filter_sizes[0],
                         kernel_size=[kernel_sizes[0], kernel_sizes[0]],
                         strides=strides, padding='SAME')
    X = tf.nn.relu(X)
    X = tf.layers.conv2d(inputs=X, filters=filter_sizes[1],
                         kernel_size=[kernel_sizes[1], kernel_sizes[1]],
                         strides=strides, padding='SAME')
    X = tf.nn.relu(X)
    return X


def softmax(input):
    k = tf.exp(input - 3)
    k = tf.reduce_sum(k, 3, True)
    ouput = tf.exp(input - 3) / k
    return ouput


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def l2_norm(x, eps=1e-12):
    return x/(tf.reduce_sum(x ** 2) ** 0.5 + eps)


def spectrum_norm(w, iter=1):
    w_shape = w.shape.as_list()
    w_new_shape = [np.prod(w_shape[:-1]), w_shape[-1]]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped')
    u = tf.get_variable("u_vec", [w_new_shape[0], 1],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    v_ = None
    for _ in range(iter):
        v_ = l2_norm(tf.matmul(tf.transpose(w_reshaped), u_))
        u_ = l2_norm(tf.matmul(w_reshaped, v_))
    u_final = tf.identity(u_, name='u_final')
    v_final = tf.identity(v_, name='v_final')

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)
    sigma = tf.matmul(tf.matmul(tf.transpose(u_final),
                                w_reshaped), v_final, name="est_sigma")
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def convSN2d(input_l, out_dim, kernel, stride, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",
                            shape=[kernel, kernel, input_l.get_shape()
                                   [-1], out_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            "b", [out_dim], initializer=tf.constant_initializer(0.))
        X = tf.nn.conv2d(input_l, spectrum_norm(w),
                         strides=[1, stride, stride, 1], padding='SAME')
        X = tf.nn.bias_add(X, b, name="c_add_b")
        return X
