import tensorflow as tf
import numpy


class ModelBase(object):
    def __init__(self, mode='train', options=None, dataset=None):
        assert (mode in ['train', 'load_train', 'predict'])
        if options == None:
            raise ValueError
        self.dataset = dataset
        self.mode = mode
        self.options = options
        self.ckpt_name = options['ckpt_name']
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def input_ops(self):
        raise NotImplementedError

    def build_model(self, input_tensor):
        raise NotImplementedError

    def train_ops(self):
        raise NotImplementedError

    def ground_truth_ops(self):
        raise NotImplementedError

    def logger(self):
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all(name='logger')
        return merged

    def loss_ops(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save(self):
        self.saver = tf.train.Saver()
        self.saver.save(self.sess, "./ckpt/{}".format(self.ckpt_name))

    def load(self):
        # assert self.mode in ['load_train', 'test']
        # path = tf.train.latest_checkpoint('./ckpt', self.ckpt_name)
        self.loader = tf.train.import_meta_graph("./ckpt/{}.meta".format(self.ckpt_name))
        self.loader.restore(sess=self.sess, save_path="./ckpt/{}".format(self.ckpt_name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
