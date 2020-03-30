import tensorflow as tf
import numpy

import model.model_base_regression
from Lib.misc import weight_variable, bias_variable, piecewise_rate
from tensorflow.contrib import slim

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"


class ModelDnnRegression(model.model_base_regression.ModelBaseRegression):
    def build_model(self, input_tensor):
        keep_prob_tensor = tf.placeholder(tf.float32, name="keep_prob_tensor")
        size_flat = (self.CYCLE + 1) * self.STATE * self.MEASURE

        nn_shape = [size_flat, 1024, 512, 64, 32, 8]
        # func1 layer #
        w_fc1 = weight_variable([nn_shape[0], nn_shape[1]])
        b_fc1 = bias_variable([nn_shape[1]])
        h_fc1 = tf.nn.relu(tf.matmul(input_tensor, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_tensor)
        # func2 layer #
        w_fc2 = weight_variable([nn_shape[1], nn_shape[2]])
        b_fc2 = bias_variable([nn_shape[2]])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_tensor)
        # func3 layer #
        w_fc3 = weight_variable([nn_shape[2], nn_shape[3]])
        b_fc3 = bias_variable([nn_shape[3]])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob_tensor)
        # func4 layer #
        w_fc4 = weight_variable([nn_shape[3], nn_shape[4]])
        b_fc4 = bias_variable([nn_shape[4]])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, w_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob_tensor)
        # func5 layer #
        w_fc5 = weight_variable([nn_shape[4], nn_shape[5]])
        b_fc5 = bias_variable([nn_shape[5]])
        prediction = tf.add(tf.matmul(h_fc4_drop, w_fc5), b_fc5, name='prediction')
        return prediction
