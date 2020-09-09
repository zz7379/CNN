import tensorflow as tf
import numpy
import model.model_base_regression
from Lib.misc import weight_variable, bias_variable, piecewise_rate
from tensorflow.contrib import slim

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"


class ModelCnnRegression(model.model_base_regression.ModelBaseRegression):
    def build_model(self, input_tensor):
        x_image = tf.reshape(input_tensor, [-1, (self.CYCLE + 1), self.STATE, self.MEASURE])
        keep_prob_tensor = tf.placeholder(tf.float32, name="keep_prob_tensor")
        with tf.name_scope('conv_1'):
            #w_conv1 = weight_variable([1, self.STATE, 18, 32])
            w_conv1 = weight_variable([3, 3, self.MEASURE, 16])
            b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding="SAME")
            h_add1 = tf.add(h_conv1, b_conv1, name="conv1_add")
            h_active1 = tf.nn.relu(h_add1)
            h_pool1 = tf.nn.max_pool(h_active1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding="VALID", name="conv1_out")  # output size 14*14*32  3
        with tf.name_scope('conv_2'):
            w_conv2 = weight_variable([3, 3, 16, 32])  # kernel 5*5, in size 32, out size 64
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")
            h_add2 = tf.add(h_conv2, b_conv2, name="conv2_add")
            h_active2 = tf.nn.relu(h_add2)
            h_pool2 = tf.nn.max_pool(h_active2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding="VALID", name="conv2_out")  # output size 7*7*64
        with tf.name_scope('fc_1'):
            shape = h_pool2.get_shape().as_list()
            size_flat = shape[1] * shape[2] * shape[3]
            # print("size_flat = {}".format(size_flat))
            w_fc1 = weight_variable([size_flat, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, size_flat])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_tensor)
        with tf.name_scope('fc_2'):
            w_fc2 = weight_variable([1024, 1024])
            b_fc2 = bias_variable([1024])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="fc_2")
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob_tensor)
        with tf.name_scope('fc_3'):
            w_fc3 = weight_variable([1024, 8])
            b_fc3 = bias_variable([8])
        prediction = tf.add(tf.matmul(h_fc2_drop, w_fc3), b_fc3, name='prediction')
        return prediction
