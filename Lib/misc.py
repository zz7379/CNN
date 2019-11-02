import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def piecewise_rate(it, learning_step, learning_rate):
    assert learning_step.__len__() == learning_rate.__len__() - 1
    section = learning_step.__len__()
    learning_step = [0] + learning_step
    for i in range(section):
        if learning_step[i] <= it < learning_step[i+1]:
            return learning_rate[i]
    return learning_rate[section]
