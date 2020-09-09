import numpy as np
from model import model_cnn_regression
from model import model_base_regression
import dataset as ds
from Lib.mat2npy import mat2npy
import tensorflow as tf
import Lib.misc
from model import model_dnn
import data_overview
import gc

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\train"
np.set_printoptions(threshold=100)
DEFAULT_OPTIONS = {'DEBUG': 0,
                   'CYCLE': 60,
                   'MEASURE': 18,
                   'STATE': 25,
                   'KEEP_PROB': 0.8,
                   'ckpt_name': 'CNN_1',
                   'BATCH_SIZE': 32,
                   'TEST_BATCH_SIZE': 10,
                   'MAX_ITERATION': 6000,
                   'learning_step': [1000, 1500, 2000, 2500, 4000],
                   'learning_rate': [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]}

MASK = [1, 1,    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
options = DEFAULT_OPTIONS
options["MEASURE"] = 7
#MASK = [0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#MASK = [1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 测量参数设置 H/12000, Ma, Tt21 Pt21 Tt3 Pt3 Tt4 Pt4 Tt44 Pt44 Tt5 Pt5 Tt9 Pt9 NH W21 Wf F


def read_images_mat(path):
    data = np.load(path)
    data[np.isnan(data)] = 0
    data_max, data_min, data0 = np.zeros((options["MEASURE"])), np.zeros((options["MEASURE"])), np.zeros((data.shape[0],data.shape[1],data.shape[2],options["MEASURE"]))
    for ii in range(options["MEASURE"]):
        if MASK[ii] == 0:
            data[:, :, :, ii] = 0
        data_max[ii] = data[:, :, :, ii].max()
        data_min[ii] = data[:, :, :, ii].min()

    for ii in range(options["MEASURE"]):
        if MASK[ii]:
            data0[:, :, :, ii] = (data[:, :, :, ii] - data_min[ii]) / (data_max[ii] - data_min[ii])
    return data0

images = read_images_mat(r"./npy/images.npy") # 30台
labels = np.load(r"./npy/labels.npy")
test_images = read_images_mat(r"./npy/test_images.npy")# 1台
test_labels = np.load(r"./npy/test_labels.npy")

images2 = read_images_mat(r"./npy/images.npy") # 30台
labels2 = np.load(r"./npy/labels.npy")

finetune_images = read_images_mat(r"./npy/finetune_images.npy")
finetune_labels = np.load(r"./npy/finetune_labels.npy")
finetune_test_images = read_images_mat(r"./npy/finetune_test_images.npy")
finetune_test_labels = np.load(r"./npy/finetune_test_labels.npy")

finetune_whole_images = np.concatenate((finetune_images, finetune_test_images), axis=0)
finetune_whole_labels = np.concatenate((finetune_labels, finetune_test_labels), axis=0)


# ------------------------------------------------------------
images30 = [] #循环数为30时的所有发动机的数据
labels30 = []
for i in range(30):
    for j in range(10):
        if j == 0 & i == 0:
            images30 = np.reshape(images[30 * 10 + i * 610 + j], (1, 61, 25, options["MEASURE"]))
            labels30 = np.reshape(labels[30 * 10 + i * 610 + j], (1, 8))
        else:
            images30 = np.concatenate((images30, np.reshape(images[30 * 10 + i * 610 + j], (1, 61, 25, options["MEASURE"]))))
            labels30 = np.concatenate((labels30, np.reshape(labels[30 * 10 + i * 610 + j], (1, 8))))
test_images30 = [] #循环数为30的目标发动机的数据
test_labels30 = []
for j in range(10):
    if j == 0:
        test_images30 = np.reshape(test_images[30 * 10 + j], (1, 61, 25, options["MEASURE"]))
        test_labels30 = np.reshape(test_labels[30 * 10 + j], (1, 8))
    else:
        test_images30 = np.concatenate((test_images30, np.reshape(test_images[30 * 10 + j], (1, 61, 25, options["MEASURE"]))))
        test_labels30 = np.concatenate((test_labels30, np.reshape(test_labels[30 * 10 + j], (1, 8))))

print(np.average(test_labels30, 0))

# ---------------------------train------------------------

dataset = ds.read_data_sets(images, labels, test_images30, test_labels30, fake_data=0)
cnn = model_cnn_regression.ModelCnnRegression(mode='start', options=options, dataset=dataset)
cnn.BATCH_SIZE = 32
cnn.TEST_BATCH_SIZE = 10
cnn.train()
# test_x = np.reshape(images2, (-1, (cnn.CYCLE + 1) * cnn.MEASURE * cnn.STATE))
# test_y = np.reshape(labels2, (-1, 8))
# pred_y = np.array(cnn.predict(test_x, test_y))
# pred_y = pred_y.reshape(labels2.shape)
#
# data_overview.dataset_overview(labels2)
#
#
# images_merged = np.concatenate((images, images2), 0)
# labels_merged = np.concatenate((labels, pred_y), 0)
#
# del images, labels, dataset, images2
# gc.collect()
#
# dataset_unlabeled = ds.read_data_sets(images_merged, labels_merged, test_images30, test_labels30, fake_data=0)
#
# cnn.dataset = dataset_unlabeled
# cnn.mode = "load"
# cnn.train()