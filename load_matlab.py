from Lib.mat2npy import mat2npy
import numpy as np


test_images = mat2npy(r'./mat/dataset_aug_test.mat', 'dataset_error_aug')
test_labels = mat2npy(r'./mat/dataset_aug_test.mat', 'label_error_aug')
np.save(r"./npy/test_images.npy", test_images)
np.save(r"./npy/test_labels.npy", test_labels)

images = mat2npy(r'./mat/dataset_aug.mat', 'dataset_error_aug')
labels = mat2npy(r'./mat/dataset_aug.mat', 'label_error_aug')
np.save(r"./npy/images.npy", test_images)
np.save(r"./npy/labels.npy", test_labels)

print(images.shape)
print(labels.shape)
print(test_images.shape)
print(test_labels.shape)
