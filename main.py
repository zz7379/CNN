import numpy as np
from model import model_cnn_regression
from model import model_base_regression
import dataset as ds
from Lib.mat2npy import mat2npy
import tensorflow as tf
import Lib.misc
from model import model_dnn

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\train"
np.set_printoptions(threshold=1000)
DEFAULT_OPTIONS = {'DEBUG': 0,
                   'CYCLE': 60,
                   'MEASURE': 18,
                   'STATE': 25,
                   'KEEP_PROB': 0.8,
                   'ckpt_name': 'CNN_1',
                   'BATCH_SIZE': 32,
                   'TEST_BATCH_SIZE': 10,
                   'MAX_ITERATION': 20000,
                   'learning_step': [1000, 1500, 2000, 2500, 4000],
                   'learning_rate': [1e-4, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]}

options = DEFAULT_OPTIONS


# 测量参数设置 H/12000, Ma, Tt21 Pt21 Tt3 Pt3 Tt4 Pt4 Tt44 Pt44 Tt5 Pt5 Tt9 Pt9 NH W21 Wf F
MASK = [1, 1,    1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]

def read_images_mat(path):
    data = np.load(path)
    data_max, data_min, data0  = np.array([]), np.array([]), np.array([])
    for ii in data.shape[3]:
        if MASK[ii] == 0:
            data[:, :, :, ii] = 0
        data_max[ii] = max(data[:, :, :ii])
        data_min[ii] = min(data[:, :, :ii])

    for ii in data.shape[3]:
        data0[:, :, :, ii] = (data[:, :, :, ii] - data_min[ii]) / (data_max[ii] - data_min[ii])

    return data0



#MASK = [0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#MASK = [1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


images = read_images_mat(r"./npy/images.npy") # 30台
labels = np.load(r"./npy/labels.npy")
test_images = read_images_mat(r"./npy/test_images.npy")# 1台
test_labels = np.load(r"./npy/test_labels.npy")

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

# 归一化 高度,Ma 参数 0~12000 -> 0~0.05  0~0.8->0~0.08       发动机1 循环1 误差1
#                                                           发动机1 循环1 误差2。。。。。
#                                                            发动机1 循环2 误差1
test_images[:, :, :, 0] = test_images[:, :, :, 0] / 240000
images[:, :, :, 0] = images[:, :, :, 0] / 240000
test_images[:, :, :, 1] = test_images[:, :, :, 1] / 10
images[:, :, :, 1] = images[:, :, :, 1] / 10


finetune_images = np.load(r"./npy/finetune_images.npy")
finetune_labels = np.load(r"./npy/finetune_labels.npy")
finetune_test_images = np.load(r"./npy/finetune_test_images.npy")
finetune_test_labels = np.load(r"./npy/finetune_test_labels.npy")
#test_labels[-10:, :] = test_labels[-20:-10, :]

for i in range(18):
    if MASK[i] == 0:
        finetune_images[:, :, :, i] = 0
        finetune_test_images[:, :, :, i] = 0

# 归一化 高度,Ma 参数 0~12000 -> 0~0.05  0~0.8->0~0.08
finetune_test_images[:, :, :, 0] = finetune_test_images[:, :, :, 0] / 240000 # 全寿命
finetune_images[:, :, :, 0] = finetune_images[:, :, :, 0] / 240000 # 只有前3000寿命循环
finetune_test_images[:, :, :, 1] = finetune_test_images[:, :, :, 1] / 10
finetune_images[:, :, :, 1] = finetune_images[:, :, :, 1] / 10

finetune_whole_images = np.concatenate((finetune_images, finetune_test_images), axis=0)
finetune_whole_labels = np.concatenate((finetune_labels, finetune_test_labels), axis=0)

# options['learning_rate'] = [0,0,0,0,0,0]

# ------------------------------------------------------------
images30 = []
labels30 = []
for i in range(30):
    for j in range(10):
        if images30 == []:
            images30 = np.reshape(images[30 * 10 + i * 610 + j], (1, 61, 25, 18))
            labels30 = np.reshape(labels[30 * 10 + i * 610 + j], (1, 8))
        else:
            images30 = np.concatenate((images30, np.reshape(images[30 * 10 + i * 610 + j], (1, 61, 25, 18))))
            labels30 = np.concatenate((labels30, np.reshape(labels[30 * 10 + i * 610 + j], (1, 8))))
test_images30 = []
test_labels30 = []
for j in range(10):
    if test_images30 == []:
        test_images30 = np.reshape(test_images[30 * 10 + j], (1, 61, 25, 18))
        test_labels30 = np.reshape(test_labels[30 * 10 + j], (1, 8))
    else:
        test_images30 = np.concatenate((test_images30, np.reshape(test_images[30 * 10 + j], (1, 61, 25, 18))))
        test_labels30 = np.concatenate((test_labels30, np.reshape(test_labels[30 * 10 + j], (1, 8))))
options['MAX_ITERATION'] = 4000
print(np.average(test_labels30, 0))

# -----------------------train--------------------------------
#
dataset = ds.read_data_sets(images, labels, test_images30, test_labels30, fake_data=0)

cnn = model_cnn_regression.ModelCnnRegression(mode='load',options=options, dataset=dataset)
cnn.BATCH_SIZE = 10
cnn.TEST_BATCH_SIZE = 10
#cnn.train()


cnn.KEEP_PROB = 0.7
cnn.learning_rate = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9]

test_x = np.reshape(test_images30[2], (-1, (cnn.CYCLE + 1) * cnn.MEASURE * cnn.STATE))
test_y = np.reshape(test_labels30[2], (1, 8))
pred_y = cnn.predict(test_x, test_y)
print(test_y)
print(pred_y)


# -----------------------semi supervised-----------------


# --------------------finetune--------------------------------
# best_match = [i for i in range(10)]
# for i in range(np.shape(best_match)[0]):
#     if i == 0:
#         best_match_images = images[best_match[0] * 610 + 300: best_match[0] * 610 + 310]
#         best_match_labels = labels[best_match[0] * 610 + 300: best_match[0] * 610 + 310]
#     else:
#         best_match_images = np.concatenate((best_match_images,
#                                            images[best_match[i] * 610 + 300: best_match[i] * 610 + 310]), axis=0)
#         best_match_labels = np.concatenate((best_match_labels,
#                                            labels[best_match[i] * 610 + 300: best_match[i] * 610 + 310]), axis=0)
#
# dataset_finetune = ds.read_data_sets(best_match_images, best_match_labels, test_images30, test_labels30, fake_data=0)
# cnn = model_cnn_regression.ModelCnnRegression(mode='train', options=options, dataset=dataset_finetune)
#
# cnn.KEEP_PROB = 0.8
# cnn.learning_rate = [1e-6, 5e-7, 1e-7, 5e-8, 1e-8, 5e-9]
# cnn.finetune()
# --------------------------cycle_match-------------------------------
# cycle_score = np.zeros(61)
# feature_test_images30 = []
# for i in range(10):
#     if feature_test_images30 == []:
#         feature_test_images30 = cnn.feature_extract(np.reshape(test_images30[i], (1, -1)))
#     else:
#         feature_test_images30 = np.concatenate((feature_test_images30, cnn.feature_extract(np.reshape(test_images30[i], (1, -1)))))
# for i in range(np.shape(images)[0]):
#     feature_temp = cnn.feature_extract(np.reshape(images[i], (1, -1)))
#     sum_cor = 0
#     for j in range(10):
#         sum_cor += Lib.misc.vectorial_angle(feature_test_images30[j], feature_temp.reshape(-1))
#     cycle_score[i % 610 // 10] += sum_cor
# print(np.argsort(cycle_score)[::-1])

# --------------------------engine_match-------------------------------
cycle_score = np.zeros(30)
feature_test_images30 = []
for i in range(10):
    if feature_test_images30 == []:
        feature_test_images30 = cnn.feature_extract(np.reshape(test_images30[i], (1, -1)))
    else:
        feature_test_images30 = np.concatenate((feature_test_images30, cnn.feature_extract(np.reshape(test_images30[i], (1, -1)))))
for i in range(np.shape(images)[0]):
    feature_temp = cnn.feature_extract(np.reshape(images[i], (1, -1)))
    sum_cor = 0
    for j in range(10):
        sum_cor += Lib.misc.vectorial_angle(feature_test_images30[j], feature_temp.reshape(-1))
    cycle_score[i // 610] += sum_cor
print(np.argsort(cycle_score)[::-1])

# --------------------feature_extract_for_finetune-------------------------
#
# CYC_NUM = options["CYCLE"]
# DIAG_CYC = 30
# feature_vector = []
# feature_finetune_vector = []
# for i in range(30):
#     for j in range(10):
#         feature_images = cnn.feature_extract(images[DIAG_CYC * 10 + i * 610 + j])
#         if feature_vector == []:
#             feature_vector = np.reshape(feature_images, (1, -1))
#         else:
#             feature_vector = np.concatenate((feature_vector, np.reshape(feature_images, (1, -1))))
#
# for j in range(10):
#     feature_finetune_images = cnn.feature_extract(test_images[DIAG_CYC * 10 + j])
#     if feature_finetune_vector == []:
#         feature_finetune_vector = np.reshape(feature_finetune_images, (1, -1))
#     else:
#         feature_finetune_vector = np.concatenate((feature_finetune_vector, np.reshape(feature_finetune_images, (1, -1))))
#
# correlation_angle = list([])
# correlation_distance = list([])
# sum_correlation_angle = 0
# sum_correlation_distance = 0
# for i in range(0, feature_vector.shape[0]):
#     for j in range(10):
#         sum_correlation_angle += Lib.misc.vectorial_angle(feature_vector[i], feature_finetune_vector[j])
#         sum_correlation_distance += Lib.misc.euclidean_distance(feature_vector[i], feature_finetune_vector[j])
#     if i % 10 == 9:
#         correlation_angle.append(sum_correlation_angle / 100)
#         sum_correlation_angle = 0
#         correlation_distance.append(sum_correlation_distance / 100)
#         sum_correlation_distance = 0
#
# print("Correlation_angle = ", correlation_angle)
# print("Index = ", np.argsort(correlation_angle)[::-1])
# print("Correlation_distance = ", correlation_distance)
# print("Index = ", np.argsort(correlation_distance))
# # best match 11,9, 16,7, 15,22,14,12,1, 13,  6,  8, 10, 18, 28, 25, 23,4, 27, 29, 20,  3, 26, 24,  2, 19, 21, 17,0,  5

# --------------------feature extract for semi-supervised learning-------------------------
# feature_ssr_vector = []
# for i in range(np.shape(images)[0]):
#     print(i)
#     feature_images = cnn.feature_extract(images[i])
#     if i == 0:
#         feature_ssr_vector = np.reshape(feature_images, (1, -1))
#     else:
#         feature_ssr_vector = np.concatenate((feature_ssr_vector, np.reshape(feature_images, (1, -1))))
#
# np.save(r"./npy/image_conv1_feature.npy", feature_ssr_vector)