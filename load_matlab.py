from Lib.mat2npy import mat2npy
import numpy as np
'''[Engine0MeasureError0~Engine0MeasureError1~....., cycle, state, measurement]'''


images = mat2npy(r'./mat/dataset_aug.mat', 'dataset_error_aug')
labels = mat2npy(r'./mat/dataset_aug.mat', 'label_error_aug')
# np.save(r"./npy/images2.npy", images)
# np.save(r"./npy/labels2.npy", labels)

images2 = mat2npy(r'./mat/dataset2_aug.mat', 'dataset_error_aug')
labels2 = mat2npy(r'./mat/dataset2_aug.mat', 'label_error_aug')
# np.save(r"./npy/images2.npy", images2)
# np.save(r"./npy/labels2.npy", labels2)

images12 = np.concatenate((images, images2))
labels12 = np.concatenate((labels, labels2))
np.save(r"./npy/images12.npy", images12)
np.save(r"./npy/labels12.npy", labels12)
#
# test_images = mat2npy(r'./mat/dataset_aug_test.mat', 'dataset_error_aug')
# test_labels = mat2npy(r'./mat/dataset_aug_test.mat', 'label_error_aug')
# np.save(r"./npy/test_images.npy", test_images)
# np.save(r"./npy/test_labels.npy", test_labels)
#
#
#
# t_images = mat2npy(r'./mat/dataset_finetune_test.mat', 'dataset_error_aug')
# t_labels = mat2npy(r'./mat/dataset_finetune_test.mat', 'label_error_aug')
# DIAG_CYC = 31
# finetune_test_images = t_images[310:610, :, :, :]
# finetune_test_labels = t_labels[310:610, :]
# finetune_images = t_images[0:310, :, :, :]
# finetune_labels = t_labels[0:310, :]
# np.save(r"./npy/finetune_test_images.npy", finetune_test_images)
# np.save(r"./npy/finetune_test_labels.npy", finetune_test_labels)
# np.save(r"./npy/finetune_images.npy", finetune_images)
# np.save(r"./npy/finetune_labels .npy", finetune_labels)








#
# finetune_test_images = test_images[31:61, :, :, :]
# finetune_test_labels = test_labels[31:61, :]
# finetune_images = test_images[0:31, :, :, :]
# finetune_labels = test_labels[0:31, :]
# print(finetune_test_images.shape)
# print(finetune_test_labels.shape)
# for i in range(1, 10):
#     finetune_test_images = np.concatenate((finetune_test_images, test_images[i * 61 + 31:i * 61 + 61, :, :, :]))
#     finetune_test_labels = np.concatenate((finetune_test_labels, test_labels[i * 61 + 31:i * 61 + 61, :]))
#     finetune_images = np.concatenate((finetune_images, test_images[i * 61:i * 61 + 31, :, :, :]))
#     finetune_labels = np.concatenate((finetune_labels, test_labels[i * 61:i * 61 + 31, :]))
#     print(finetune_test_images.shape)
#     print(finetune_test_labels.shape)