import numpy as np
from model import model_cnn_regression
import dataset as ds
from Lib.mat2npy import mat2npy
import tensorflow as tf
import Lib.misc
from model import model_dnn

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"
np.set_printoptions(threshold=1000)
DEFAULT_OPTIONS = {'DEBUG': 0,
                   'CYCLE': 60,
                   'MEASURE': 18,
                   'STATE': 25,
                   'KEEP_PROB': 0.7,
                   'ckpt_name': 'CNN_1',
                   'BATCH_SIZE': 64,
                   'TEST_BATCH_SIZE': 20,
                   'MAX_ITERATION': 20000,
                   'learning_step': [1000, 1500, 2000, 2500, 5000],
                   'learning_rate': [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]}

options = DEFAULT_OPTIONS

with open('result.txt', 'a') as file_handle:  # 保存结果
    file_handle.write("cnn+finetune")
    file_handle.write('\n')

# 测量参数设置 H/12000, Ma, Tt21 Pt21 Tt3 Pt3 Tt4 Pt4 Tt44 Pt44 Tt5 Pt5 Tt9 Pt9 NH W21 Wf F
MASK = [1, 1,    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
#MASK = [0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#MASK = [1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

options['MAX_ITERATION'] = 8000
images = np.load(r"./npy/images.npy")
labels = np.load(r"./npy/labels.npy")
test_images = np.load(r"./npy/test_images.npy")
test_labels = np.load(r"./npy/test_labels.npy")

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

# 归一化 高度,Ma 参数 0~12000 -> 0~0.05  0~0.8->0~0.08
test_images[:, :, :, 0] = test_images[:, :, :, 0] / 240000
images[:, :, :, 0] = images[:, :, :, 0] / 240000
test_images[:, :, :, 1] = test_images[:, :, :, 1] / 10
images[:, :, :, 1] = images[:, :, :, 1] / 10


ordered_test_images = test_images.reshape(test_images.shape[0], -1)
ordered_test_labels = test_labels

# print('images\n'*5, test_images)
# print('labels\n'*5, test_labels)

dataset = ds.read_data_sets(images, labels, test_images, test_labels, fake_data=0)

cnn = model_cnn_regression.ModelCnnRegression(mode='train', options=options, dataset=dataset)
cnn.train()

# dnn = model_dnn.ModelDnnRegression(mode='train', options=options, dataset=dataset)
# dnn.train()
# test_rmse=0.00066437 test_rmse=0.00066437

'''
cnn.pred_input_x = ordered_test_images
cnn.pred_input_y = ordered_test_labels
cnn.mode = 'predict'

print('res '*50, cnn.run())
print('ans '*50, ordered_test_labels)
'''


# finetune

images = np.load(r"./npy/finetune_images.npy")
labels = np.load(r"./npy/finetune_labels.npy")
test_images = np.load(r"./npy/finetune_test_images.npy")
test_labels = np.load(r"./npy/finetune_test_labels.npy")
# test_labels[-10:, :] = test_labels[-20:-10, :]

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

# 归一化 高度,Ma 参数 0~12000 -> 0~0.05  0~0.8->0~0.08
test_images[:, :, :, 0] = test_images[:, :, :, 0] / 240000
images[:, :, :, 0] = images[:, :, :, 0] / 240000
test_images[:, :, :, 1] = test_images[:, :, :, 1] / 10
images[:, :, :, 1] = images[:, :, :, 1] / 10

options['learning_rate'] = [1e-8, 5e-9, 1e-9, 5e-10, 1e-10, 5e-11]
#cnn.load()
dataset_finetune = ds.read_data_sets(images, labels, test_images, test_labels, fake_data=0)
cnn.dataset = dataset_finetune
cnn.finetune()


