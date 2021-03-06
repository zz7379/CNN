# coding:utf-8
import tensorflow as tf
import numpy
from old import input_data

#print(device_lib.list_local_devices())

CYCLE = 29
MEASURE = 18
STATE = 25
MASK = [1, 1,        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# H Ma Tt21 Pt21 Tt3 Pt3 Tt4 Pt4 Tt44 Pt44 Tt5 Pt5 Tt9 Pt9 NH W21 Wf F

images = numpy.load(r"./npy/images.npy")
labels = numpy.load(r"./npy/labels.npy")
test_images = numpy.load(r"./npy/test_images.npy")
test_labels = numpy.load(r"./npy/test_labels.npy")

for i in range(18):
    if MASK[i] == 0:
        images[:, :, :, i] = 0
        test_images[:, :, :, i] = 0

max1 = images.max()
max2 = test_images.max()
maxi = max1 if(max1 > max2) else max2
min1 = images.min()
min2 = test_images.min()
mini = min1 if(min1 < min2) else min2
images = (images - mini) / (maxi - mini)
test_images = (test_images - mini) / (maxi - mini)
print(images.shape)
dataset = input_data.read_data_sets(images, labels, test_images, test_labels, one_hot=False)



def compute_accuracy(v_xs, v_ys):
    #y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    y_pre = tf.get_default_graph.get_tensor_by_name("prediction:0")
    sq = tf.square(y_pre - v_ys)
    mse = tf.reduce_mean(sq)
    result = sess.run(mse, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] =1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")  # padding="SAME"用零填充边界


def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="VALID")


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, (CYCLE + 1) * MEASURE * STATE])  # 8*stride
ys = tf.placeholder(tf.float32, [None, 8])
# 定义dropout的输入，解决过拟合问题
keep_prob = tf.placeholder(tf.float32)
#x_image = tf.reshape(xs, [-1, 1, window_size, 8])
# print(x_image.shape) #[n_samples, 28,28,1]

## funcl layer ##
W_fc1 = weight_variable([(CYCLE + 1) * MEASURE * STATE, 1024])
b_fc1 = bias_variable([1024])
# [n_samples,7,7,64]->>[n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## func2 layer ##
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])
prediction = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2,name='prediction')
## acc ##
sq = tf.square(prediction - ys)
mse = tf.reduce_mean(sq)
tf.summary.scalar('mse', mse)
# #################优化神经网络##################################
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(rate).minimize(mse)
sess = tf.Session()
# important step

# ##################Tensorboar_Summary############
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(r"C:\tflog\train", sess.graph)
test_writer = tf.summary.FileWriter(r"C:\tflog\test")
sess.run(tf.initialize_all_variables())
#writer = tf.summary.FileWriter(r"C:\tflog", sess.graph)
# console  C:\Users\70951\AppData\Roaming\Python\Python37\Scripts\tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"



#################优化神经网络##################################
BATCH_SIZE = 50
test_batch = dataset.test.next_batch(200)
xs_test = test_batch[0].reshape(-1, (CYCLE + 1) * MEASURE * STATE)
ys_test = test_batch[1]
acc_test = numpy.zeros(200)
acc_train = numpy.zeros(200)
acci = 0
variable_summaries(xs_test)

for epoch in range(20000):
    if epoch < 500:
        ra = 1e-5
    elif epoch < 1000:
        ra = 5e-5
    elif epoch < 2000:
        ra = 1e-6
    elif epoch < 4000:
        ra = 3e-6
    elif epoch < 8000:
        ra = 1e-7
    else:
        ra = 1e-7
    batch = dataset.train.next_batch(BATCH_SIZE)
    xs_train = batch[0].reshape(-1, (CYCLE + 1) * MEASURE * STATE)
    ys_train = batch[1]


    summary, _ = sess.run([merged, train_step], feed_dict={xs: xs_train, ys: ys_train, keep_prob: 0.5, rate: ra})
    #train_writer.add_summary(summary, epoch)
    if epoch % 100 == 0:
        # print(sess.run(prediction,feed_dict={xs: batch_xs}))
        #summary, output = sess.run([merged, 'softmax:0'], {xs: xs_test,keep_prob: 0.5})
        #assert not(numpy.isnan(output.any()))
        summary, acc_train[acci] = sess.run([merged, mse], {xs: xs_train, ys: ys_train, keep_prob: 1, rate: ra})
        train_writer.add_summary(summary, epoch)
        summary, acc_test[acci], y_pred = sess.run([merged, mse, prediction], {xs: xs_test, ys: ys_test, keep_prob: 1, rate: ra})
        #acc_test[acci] = compute_accuracy(xs_test,ys_test)
        #acc_train[acci] = compute_accuracy(xs_train, ys_train)
        acci = acci + 1
        print(epoch, " test_mse={:.8f}  train_mse={:.8f}  test_rmse={:.8f}  train_rmse={:.8f}  ".format(acc_test[acci-1], acc_train[acci-1], numpy.sqrt(acc_test[acci-1]), numpy.sqrt(acc_train[acci-1])))
        test_writer.add_summary(summary, epoch)

# console  C:\Users\70951\AppData\Roaming\Python\Python37\Scripts\tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"
print(acc_test)
import numpy as np
from matplotlib import pyplot as plt

plt.title("Epoch-MSE plot")
plt.xlabel("Epoch")
plt.ylabel("MSE")
x=np.arange(50)
plt.plot(x, acc_test)
plt.show()








