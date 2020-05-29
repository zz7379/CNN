import tensorflow as tf
import numpy
from model import model_base
from Lib.misc import piecewise_rate


# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"


class ModelBaseRegression(model_base.ModelBase):
    def __init__(self, mode='train', options=None, dataset=None, pred_input_x=None, pred_input_y=None):
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
        self.loss = self.loss_ops()
        self.pred_input_x = pred_input_x
        self.pred_input_y = pred_input_y
        self.train_writer = tf.summary.FileWriter(r"C:\tflog\train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(r"C:\tflog\test")
        self.rate = tf.placeholder(tf.float32, name='rate')
        self.trainer = self.train_ops()
        self.sess.run(tf.initialize_all_variables())
        foo = self.logger()
        # self.global_step = slim.get_or_create_global_step
        # self.tensor_learning_rate = tf.train.piecewise_constant(
        #     self.global_step,
        #     [tf.cast(v, tf.int64) for v in self.learning_step],
        #     self.learning_rate)

    def input_ops(self):
        x = tf.placeholder(tf.float32, [None, (self.CYCLE + 1) * self.MEASURE * self.STATE], name='x')  # 8*stride
        return x

    def build_model(self, input_tensor):
        raise NotImplementedError

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
        train_op = optimizer.apply_gradients(grads_and_vars)  # 梯度截断
        return train_op

    def train(self, trainer=None):
        if trainer == None:
            trainer = self.trainer
        self.sess.run(tf.initialize_all_variables())
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
            res = 0
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
                res = "test_mse={:.8f}  train_mse={:.8f}  test_rmse={:.8f}  train_rmse={:.8f}".format(
                    acc_test[acc_index - 1], acc_train[acc_index - 1], numpy.sqrt(acc_test[acc_index - 1]),
                    numpy.sqrt(acc_train[acc_index - 1]))
            if it % 2000 == 0 & self.options["DEBUG"]:
                relative_error = (pred_test-y_test)/y_test
                print("relative_error = ", numpy.average(relative_error, axis=0))
                #print(pred_test, '\n', '-'*50, '\n', y_test)

        #self.save()
        with open('result.txt', 'a') as file_handle:  # 保存结果
            file_handle.write(str(res))
            file_handle.write('\n')

    def predict(self):
        x_tensor = tf.get_default_graph().get_tensor_by_name("x:0")
        y_tensor = tf.get_default_graph().get_tensor_by_name("y:0")
        pred_tensor = tf.get_default_graph().get_tensor_by_name("prediction:0")
        keep_prob_tensor = tf.get_default_graph().get_tensor_by_name("keep_prob_tensor:0")
        pred = self.sess.run([pred_tensor], {x_tensor: self.pred_input_x, y_tensor: self.pred_input_y, keep_prob_tensor: 1})
        return pred

    def finetune(self):
        trainable_vars = tf.trainable_variables()
        freeze_var_list = [t for t in trainable_vars if not t.name.startswith('fc_3')]
        optimizer = tf.train.AdamOptimizer(self.rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, var_list=freeze_var_list)
        for i, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                grads_and_vars[i] = (tf.clip_by_norm(g, 10), v)
        finetune_train_op = optimizer.apply_gradients(grads_and_vars)  # 梯度截断
        print("Freeze list:\n", freeze_var_list)
        #var_list = tf.get_default_graph().get_tensor_by_name(r"fc_3/Variable/(Variable):0")

        self.train(trainer=finetune_train_op)

