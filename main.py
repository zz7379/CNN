import numpy
from model import model_cnn_regression
import dataset
from model import model_dnn

# tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"

DEFAULT_OPTIONS = {'CYCLE': 60,
                   'MEASURE': 18,
                   'STATE': 25,
                   'ckpt_name': 'CNN_1',
                   'BATCH_SIZE': 200,
                   'TEST_BATCH_SIZE': 300,
                   'MAX_ITERATION': 200000,
                   'learning_step': [1000, 2000, 4000, 8000, 12000],
                   'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]}

options = DEFAULT_OPTIONS
MASK = [1, 1,    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

images = numpy.load(r"./npy/images.npy")
labels = numpy.load(r"./npy/labels.npy")
test_images = numpy.load(r"./npy/test_images.npy")
test_labels = numpy.load(r"./npy/test_labels.npy")

images[:, :, :, 1] = images[:, :, :, 1] / 20000

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

dataset = dataset.read_data_sets(images, labels, test_images, test_labels)
cnn = model_cnn_regression.ModelCnnRegression(mode='train', options=options, dataset=dataset)
cnn.run()
# dnn = model_dnn.ModelDnnRegression(mode='train', options=options, dataset=dataset)
# dnn.run()