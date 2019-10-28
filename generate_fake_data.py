import numpy as np

DATANUM = 1000
WIDTH = 32
HIGHT = 1
CHANNEL = 8

data = np.zeros((DATANUM * 15, WIDTH * HIGHT * CHANNEL))
label = np.zeros((DATANUM * 15, 1), dtype=np.uint8)

fake_num = 2

if fake_num == 1:
    for i in range(DATANUM):
        for j in range(10):
            data[i * 10 + j, :] = np.ones((1, WIDTH * HIGHT)) * j * j
            label[i * 10 + j] = j
elif fake_num == 2:
    for i in range(DATANUM):
        for j in range(15):
            data[i * 15 + j, :] = np.ones((1, WIDTH * HIGHT * CHANNEL)) * j * j + np.random.normal() * 0.5
            label[i * 15 + j] = j
elif fake_num == 3:
    data = np.zeros((DATANUM * 10, 28*28))
    label = np.zeros((DATANUM * 10, 1), dtype=np.uint8)
    for i in range(DATANUM):
        for j in range(10):
            data[i * 10 + j, :] = np.ones((1, 28*28)) * j
            label[i * 10 + j] = j

def disorder_images(images, labels):
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]
    perm = np.arange(num_examples)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    return images, labels


images, labels = disorder_images(data, label)
test_images, test_labels = disorder_images(data, label)
np.save("images.npy", images)
np.save("labels.npy", labels)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)
print(images,labels)