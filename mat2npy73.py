# -*- coding: utf-8 -*-
import scipy.io as sio 
import numpy as np
import h5py

def disorder_images(images, labels):
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm,:,:]
    labels = labels[perm]
    return images,labels


image_path = r'./mat/dataset_aug.mat'
image_D = h5py.File(image_path)
images = np.transpose(image_D['dataset_error_aug'])
labels = np.transpose(image_D['label_error_aug'])
images,labels =  disorder_images(images, labels)
np.save(r"./npy/images.npy",images)
np.save(r"./npy/labels.npy",labels)
print(images.shape)
print(labels.shape)

test_image_path = r'./mat/dataset_aug_test.mat'
test_image_D = h5py.File(test_image_path)
test_images = np.transpose(test_image_D['dataset_error_aug'])
test_labels = np.transpose(test_image_D['label_error_aug'])
test_images,test_labels = disorder_images(test_images, test_labels)
np.save(r"./npy/test_images.npy",test_images)
np.save(r"./npy/test_labels.npy",test_labels)
print(test_images.shape)
print(test_labels.shape)