import numpy as np

import dataset as ds

images = np.load(r"./npy/images.npy")
labels = np.load(r"./npy/labels.npy")
test_images = np.load(r"./npy/test_images.npy")
test_labels = np.load(r"./npy/test_labels.npy")

finetune_images = np.load(r"./npy/finetune_images.npy")
finetune_labels = np.load(r"./npy/finetune_labels.npy")
finetune_test_images = np.load(r"./npy/finetune_test_images.npy")
finetune_test_labels = np.load(r"./npy/finetune_test_labels.npy")

print(images.shape)
print(labels.shape)
print(test_images.shape)
print(test_labels.shape)
print(finetune_images.shape)
print(finetune_labels.shape)
print(finetune_test_images.shape)
print(finetune_test_labels.shape)
print('\n' * 3)

sum1 = 0
count1 = 0
sum2 = 0
count2 = 0
for a in labels:
    for b in a:
        sum1 += b*b
        count1 += 1

for a in test_labels:
    for b in a:
        sum2 += b*b
        count2 += 1

rmse1 = (sum1 / count1) ** 0.5
rmse2 = (sum2 / count2) ** 0.5
print("RMSE_train = {}    RMSE_test = {} \n".format(rmse1, rmse2))


print(labels[:,1])
print('\n'*3)
print(test_labels[:,1])
print('\n'*3)
print(finetune_labels[:,1])
print('\n'*3)
print(finetune_test_labels[:,1])
