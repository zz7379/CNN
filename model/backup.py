import tensorflow as tf
import numpy
from model import model_base
from Lib.misc import weight_variable, bias_variable, piecewise_rate


# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"

class ModelCnnRegression(model_base.ModelBase):
    def __init__(self, mode='train', options=None, dataset=None):
        super().__init__(mode, options, dataset)
        self.mode = mode
        self.ckpt_name = options['ckpt_name']
        self.CYCLE = options['CYCLE']
        self.MEASURE = options['MEASURE']
        self.STATE = options['STATE']
        self.KEEP_PROB = 1 if (self.mode != 'predict') else 0.5
        self.BATCH_SIZE = options['BATCH_SIZE']
        self.TEST_BATCH_SIZE = options['TEST_BATCH_SIZE']
        self.MAX_ITERATION = options['MAX_ITERATION']
        self.learning_rate = options['learning_rate']
        self.learning_step = options['learning_step']
        # self.global_step = slim.get_or_create_global_step
        # self.tensor_learning_rate = tf.train.piecewise_constant(
        #     self.global_step,
        #     [tf.cast(v, tf.int64) for v in self.learning_step],
        #     self.learning_rate)

    def input_ops(self):
        x = tf.placeholder(tf.float32, [None, (self.CYCLE + 1) * self.MEASURE * self.STATE], name='x')  # 8*stride
        return x

    def build_model(self, input_tensor):
        x_image = tf.reshape(input_tensor, [-1, (self.CYCLE + 1), self.STATE, self.MEASURE])
        w_conv1 = weight_variable([1, self.STATE, 18, 32])  # kernel 5*5, channel is 1, out size 32
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding="SAME")
        h_active1 = tf.nn.tanh(h_conv1 + b_conv1)
        h_pool1 = tf.nn.max_pool(h_active1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                                 padding="VALID")  # output size 14*14*32  3
        w_conv2 = weight_variable([1, 1, 32, 64])  # kernel 5*5, in size 32, out size 64
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")
        h_active2 = tf.nn.tanh(h_conv2 + b_conv2)
        h_pool2 = tf.nn.max_pool(h_active2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1],
                                 padding="VALID")  # output size 7*7*64
        # funcl layer #
        shape = h_pool2.get_shape().as_list()
        size_flat = shape[1] * shape[2] * shape[3]
        print("size_flat = {}".format(size_flat))
        w_fc1 = weight_variable([size_flat, 1024])
        b_fc1 = bias_variable([1024])
        # [n_samples,7,7,64]->>[n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, size_flat])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.KEEP_PROB)
        # func2 layer #
        w_fc2 = weight_variable([1024, 8])
        b_fc2 = bias_variable([8])
        prediction = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2, name='prediction')
        return prediction

    def ground_truth_ops(self):
        return tf.placeholder(tf.float32, [None, 8], name='y')

    def loss_ops(self):
        # y = tf.placeholder(tf.float32, [None, 8])
        sq = tf.square(self.build_model(self.input_ops()) - self.ground_truth_ops(), name='sqrt')
        loss = tf.reduce_mean(sq, name='loss')
        tf.summary.scalar('mse', loss)
        return loss

    def train_ops(self):
        rate = tf.placeholder(tf.float32, name='rate')
        train_op = tf.train.AdamOptimizer(rate).minimize(self.loss_ops())
        return train_op

    def run(self):
        if self.mode in ['load_train', 'predict']:
            self.load()
        else:
            trainer = self.train_ops()
            foo = self.logger()
            self.sess.run(tf.initialize_all_variables())
        self.train_writer = tf.summary.FileWriter(r"C:\tflog\train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(r"C:\tflog\test")
        if self.mode in ['load_train', 'train']:
            # logger_tensor = tf.get_default_graph().get_tensor_by_name("logger:0")
            # train_tensor = tf.get_default_graph().get_tensor_by_name("train:0")
            x_tensor = tf.get_default_graph().get_tensor_by_name("x:0")
            y_tensor = tf.get_default_graph().get_tensor_by_name("y:0")
            rate = tf.get_default_graph().get_tensor_by_name("rate:0")
            test_batch = self.dataset.test.next_batch(self.TEST_BATCH_SIZE)
            x_test = test_batch[0].reshape(-1, (self.CYCLE + 1) * self.MEASURE * self.STATE)
            y_test = test_batch[1]
            acc_test = numpy.zeros(2000)
            acc_train = numpy.zeros(2000)
            acc_index = 0

            for it in range(self.MAX_ITERATION):
                batch = self.dataset.train.next_batch(self.BATCH_SIZE)
                x_train = batch[0].reshape(-1, (self.CYCLE + 1) * self.MEASURE * self.STATE)
                y_train = batch[1]
                it_rate = piecewise_rate(it, self.learning_step, self.learning_rate)
                loss_tensor = tf.get_default_graph().get_tensor_by_name("loss:0")
                bar = self.sess.run([trainer], {x_tensor: x_train, y_tensor: y_train, rate: it_rate})
                if it % 100 == 0:
                    [acc_train[acc_index]] = self.sess.run([loss_tensor], {x_tensor: x_train, y_tensor: y_train})
                    [acc_test[acc_index]] = self.sess.run([loss_tensor], {x_tensor: x_test, y_tensor: y_test})
                    # self.test_writer.add_summary(summary, it)
                    acc_index = acc_index + 1
                    print(it, " test_mse={:.8f}  train_mse={:.8f}  test_rmse={:.8f}  train_rmse={:.8f}  ".format(
                        acc_test[acc_index - 1], acc_train[acc_index - 1], numpy.sqrt(acc_test[acc_index - 1]),
                        numpy.sqrt(acc_train[acc_index - 1])))
            else:
                # --------------coding---------------#
                raise ModuleNotFoundError
                return self.sess.run(tf.get_default_graph().get_tensor_by_name("prediction:0"),
                                     {x_tensor: x_train, y_tensor: y_train})
