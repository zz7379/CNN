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
                   'ckpt_name': 'CNN_1',
                   'BATCH_SIZE': 200,
                   'TEST_BATCH_SIZE': 300,
                   'MAX_ITERATION': 20000,
                   'learning_step': [2000, 4000, 6000, 8000, 16000],
                   'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]}

options = DEFAULT_OPTIONS
# 测量参数设置 H/12000, Ma, Tt21 Pt21 Tt3 Pt3 Tt4 Pt4 Tt44 Pt44 Tt5 Pt5 Tt9 Pt9 NH W21 Wf F
MASK = [1, 1,    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
#MASK = [1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# options['learning_step'] = [1000, 2000, 3000, 4000, 5000]
options['MAX_ITERATION'] = 25000
images = np.load(r"./npy/images.npy")
labels = np.load(r"./npy/labels.npy")
test_images = np.load(r"./npy/test_images.npy")
test_labels = np.load(r"./npy/test_labels.npy")

# 归一化 高度 参数
test_images[:, :, :, 0] = test_images[:, :, :, 0] / 12000
images[:, :, :, 0] = images[:, :, :, 0] / 12000

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

ordered_test_images = test_images.reshape(test_images.shape[0], -1)
ordered_test_labels = test_labels

# print('images\n'*5, test_images)
# print('labels\n'*5, test_labels)

dataset = ds.read_data_sets(images, labels, test_images, test_labels, fake_data=0)
cnn = model_cnn_regression.ModelCnnRegression(mode='train', options=options, dataset=dataset)
cnn.train()

# test_rmse=0.00066437 test_rmse=0.00066437
'''
cnn.pred_input_x = ordered_test_images
cnn.pred_input_y = ordered_test_labels
cnn.mode = 'predict'

print('res '*50, cnn.run())
print('ans '*50, ordered_test_labels)
'''

#dnn = model_dnn.ModelDnnRegression(mode='train', options=options, dataset=dataset)
#dnn.run()

test_images = mat2npy(r'./mat/dataset_finetune_test.mat', 'dataset_error_aug')
test_labels = mat2npy(r'./mat/dataset_finetune_test.mat', 'label_error_aug')
np.save(r"./npy/finetune_test_images.npy", test_images)
np.save(r"./npy/finetune_test_labels.npy", test_labels)

images = mat2npy(r'./mat/dataset_finetune.mat', 'dataset_error_aug')
labels = mat2npy(r'./mat/dataset_finetune.mat', 'label_error_aug')
np.save(r"./npy/finetune_images.npy", test_images)
np.save(r"./npy/finetune_labels.npy", test_labels)


# 归一化 高度 参数
test_images[:, :, :, 0] = test_images[:, :, :, 0] / 12000
images[:, :, :, 0] = images[:, :, :, 0] / 12000

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0


dataset_finetune = ds.read_data_sets(images, labels, test_images, test_labels, fake_data=0)
cnn.dataset = dataset_finetune
cnn.train()