import tensorflow as tf
import numpy
from model import model_base
from Lib.misc import piecewise_rate, print_tensor_name


# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"


class ModelBaseRegression(model_base.ModelBase):
    def __init__(self, mode='train', options=None, dataset=None):
        super().__init__(mode, options, dataset)
        self.mode = mode
        self.ckpt_name = options['ckpt_name']
        self.CYCLE = options['CYCLE']
        self.MEASURE = options['MEASURE']
        self.STATE = options['STATE']
        self.KEEP_PROB = options['KEEP_PROB']
        self.BATCH_SIZE = options['BATCH_SIZE']
        self.TEST_BATCH_SIZE = options['TEST_BATCH_SIZE']
        self.MAX_ITERATION = options['MAX_ITERATION']
        self.learning_rate = options['learning_rate']
        self.learning_step = options['learning_step']
        self.DEBUG = options['DEBUG']

        if self.mode == "train":
            self.rate = tf.placeholder(tf.float32, name='rate')
            self.loss = self.loss_ops()
            self.trainer = self.train_ops()

            trainable_vars = tf.trainable_variables()
            train_var_list = [t for t in trainable_vars if
                              t.name.startswith('fc_1') or t.name.startswith('fc_2') or t.name.startswith('fc_3')]
            optimizer = tf.train.AdamOptimizer(self.rate)
            grads_and_vars = optimizer.compute_gradients(self.loss, var_list=train_var_list)
            for i, (g, v) in enumerate(grads_and_vars):
                if g is not None:
                    grads_and_vars[i] = (tf.clip_by_norm(g, 10), v)
            self.finetune_train_op = optimizer.apply_gradients(grads_and_vars, name="finetune_trainer")  # 梯度截断
            print("Training list:\n", train_var_list)
            self.sess.run(tf.initialize_all_variables())
        else:
            self.load()

        self.train_writer = tf.summary.FileWriter(r"C:\tflog\train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(r"C:\tflog\test")
        foo = self.logger()


    def input_ops(self):
        x = tf.placeholder(tf.float32, [None, (self.CYCLE + 1) * self.MEASURE * self.STATE], name='x')  # 8*stride
        return x

    def build_model(self, input_tensor):
        # raise NotImplementedError
        return

    def ground_truth_ops(self):
        return tf.placeholder(tf.float32, [None, 8], name='y')

    def loss_ops(self):
        # y = tf.placeholder(tf.float32, [None, 8])
        pred = tf.subtract(self.build_model(self.input_ops()), self.ground_truth_ops(), name='sub')
        sq = tf.square(pred, name='square')
        loss = tf.reduce_mean(sq, name='loss')
        tf.summary.scalar('mse', loss)
        return loss

    def train_ops(self):
        optimizer = tf.train.AdamOptimizer(self.rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        for i, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                grads_and_vars[i] = (tf.clip_by_norm(g, 10), v)
        train_op = optimizer.apply_gradients(grads_and_vars, name='train')  # 梯度截断
        return train_op

    def train(self, trainer=None):
        if trainer == None: # 事实上该条件为 如果不是finetune
            trainer = self.trainer
        x_tensor = tf.get_default_graph().get_tensor_by_name("x:0")
        y_tensor = tf.get_default_graph().get_tensor_by_name("y:0")
        rate = tf.get_default_graph().get_tensor_by_name("rate:0")
        pred_tensor = tf.get_default_graph().get_tensor_by_name("prediction:0")
        acc_test = numpy.zeros(2000)
        acc_train = numpy.zeros(2000)
        acc_index = 0

        for it in range(self.MAX_ITERATION):
            test_batch = self.dataset.test.next_batch(self.TEST_BATCH_SIZE)
            x_test = test_batch[0].reshape(-1, (self.CYCLE + 1) * self.MEASURE * self.STATE)
            y_test = test_batch[1]
            batch = self.dataset.train.next_batch(self.BATCH_SIZE)
            x_train = batch[0].reshape(-1, (self.CYCLE + 1) * self.MEASURE * self.STATE)
            y_train = batch[1]
            it_rate = piecewise_rate(it, self.learning_step, self.learning_rate)
            loss_tensor = tf.get_default_graph().get_tensor_by_name("loss:0")
            keep_prob_tensor = tf.get_default_graph().get_tensor_by_name("keep_prob_tensor:0")
            bar = self.sess.run([trainer], {x_tensor: x_train, y_tensor: y_train, rate: it_rate, keep_prob_tensor: self.KEEP_PROB})
            if it % 100 == 0:
                [acc_train[acc_index]] = self.sess.run([loss_tensor], {x_tensor: x_train, y_tensor: y_train, keep_prob_tensor: 1})
                [acc_test[acc_index], pred_test] = self.sess.run([loss_tensor, pred_tensor], {x_tensor: x_test, y_tensor: y_test, keep_prob_tensor: 1})
                # self.test_writer.add_summary(summary, it)
                acc_index = acc_index + 1
                print(it, " test_mse={:.8f}  train_mse={:.8f}  test_rmse={:.8f}  train_rmse={:.8f}  ".format(
                    acc_test[acc_index - 1], acc_train[acc_index - 1], numpy.sqrt(acc_test[acc_index - 1]),
                    numpy.sqrt(acc_train[acc_index - 1])))
                with open('test_rmse.txt', 'a') as file_handle:  # 保存结果
                    file_handle.write(str(numpy.sqrt(acc_test[acc_index - 1])))
                    file_handle.write('\n')
                with open('train_rmse.txt', 'a') as file_handle:  # 保存结果
                    file_handle.write(str(numpy.sqrt(acc_train[acc_index - 1])))
                    file_handle.write('\n')
                relative_error = (pred_test-y_test)/y_test
                relative_error = numpy.where(relative_error > 10000, 0, relative_error)
                relative_error = numpy.where(relative_error < -10000, 0, relative_error)
                with open('test_relative_error.txt', 'a') as file_handle:  # 保存结果
                    file_handle.write(str(numpy.average(relative_error, axis=0)))
                    file_handle.write('\n')
            if it % 2000 == 0 & self.options["DEBUG"]:
                print("relative_error = ", numpy.average(relative_error, axis=0))
                # print(pred_test, '\n', '-'*50, '\n', y_test)
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, "./ckpt/{}".format(self.ckpt_name))
        #self.save()


    def predict(self, pred_input_x, pred_input_y):
        x_tensor = tf.get_default_graph().get_tensor_by_name("x:0")
        y_tensor = tf.get_default_graph().get_tensor_by_name("y:0")
        pred_tensor = tf.get_default_graph().get_tensor_by_name("prediction:0")
        keep_prob_tensor = tf.get_default_graph().get_tensor_by_name("keep_prob_tensor:0")
        pred = self.sess.run([pred_tensor], {x_tensor: pred_input_x, y_tensor: pred_input_y, keep_prob_tensor: 1})
        relative_error = (pred - pred_input_y) / pred_input_y
        relative_error = numpy.where(relative_error > 10000, 0, relative_error)
        relative_error = numpy.where(relative_error < -10000, 0, relative_error)
        print("relative_error = ", numpy.average(relative_error, axis=0))
        return pred

    def finetune(self):

        #var_list = tf.get_default_graph().get_tensor_by_name(r"fc_3/Variable/(Variable):0")
        self.KEEP_PROB = 1
        self.train(trainer=tf.get_default_graph().get_operation_by_name("finetune_trainer"))

    def feature_extract(self, image):
        # self.sess.run(tf.initialize_all_variables())
        image = image.reshape(1, -1)
        conv1_output_tnsr = tf.get_default_graph().get_tensor_by_name("conv_1/conv1_out:0")
        conv2_output_tnsr = tf.get_default_graph().get_tensor_by_name("conv_2/conv2_out:0")
        conv1_add_tnsr = tf.get_default_graph().get_tensor_by_name("conv_1/conv1_add:0")
        conv2_add_tnsr = tf.get_default_graph().get_tensor_by_name("conv_2/conv2_add:0")
        keep_prob_tensor = tf.get_default_graph().get_tensor_by_name("keep_prob_tensor:0")
        x_tensor = tf.get_default_graph().get_tensor_by_name("x:0")
        fc2_tnsr = tf.get_default_graph().get_tensor_by_name("fc_2/fc_2:0")
        pred_tnsr = tf.get_default_graph().get_tensor_by_name("prediction:0")
        feature = self.sess.run([fc2_tnsr], {x_tensor: image, keep_prob_tensor: 1})
        return numpy.reshape(feature, (numpy.shape(feature)[0], -1))

