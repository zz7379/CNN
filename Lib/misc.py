import tensorflow as tf
import numpy as np

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


def print_tensor_name():
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')

def vectorial_angle(x, y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    return cos_angle