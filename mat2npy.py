# -*- coding: utf-8 -*-
import scipy.io as sio 
import numpy as np


def disorder_images(images, labels):
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    return images,labels


image_path = r'./mat/dataset_aug.mat'
image_D = sio.loadmat(image_path)
images = image_D['dataset_error_aug']
labels = image_D['label_error_aug']

test_image_path = r'./mat/dataset_aug_test.mat'
test_image_D = sio.loadmat(test_image_path)
test_images = test_image_D['dataset_error_aug']
test_labels = test_image_D['label_error_aug']

print(labels.shape)





images,labels =  disorder_images(images, labels)
test_images,test_labels = disorder_images(test_images, test_labels)
np.save(r"./npy/images.npy",images)
np.save(r"./npy/labels.npy",labels)
np.save(r"./npy/test_images.npy",test_images)
np.save(r"./npy/test_labels.npy",test_labels)