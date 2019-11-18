# coding:utf-8
window_size = 32
import tensorflow as tf
import numpy
from old import input_data
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

images = numpy.load("images.npy")
labels = numpy.load("labels.npy")
test_images = numpy.load('test_images.npy')
test_labels = numpy.load('test_labels.npy')

max1 = images.max()
max2 = test_images.max()
maxi = max1 if(max1 > max2) else max2
min1 = images.min()
min2 = test_images.min()
mini = min1 if(min1 < min2) else min2
images = (images - mini) / (maxi - mini)
test_images = (test_images - mini) / (maxi - mini)

dataset = input_data.read_data_sets(images, labels, test_images, test_labels)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
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
xs = tf.placeholder(tf.float32, [None, 8*window_size])  # 8*stride
ys = tf.placeholder(tf.float32, [None, 15])
# 定义dropout的输入，解决过拟合问题
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 1, window_size, 8])
# print(x_image.shape) #[n_samples, 28,28,1]
## convl layer ##
W_conv1 = weight_variable([1, 3, 8, 32])  # kernel 5*5, channel is 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
h_pool1 = max_pool_1x2(h_conv1)  # output size 14*14*32  3
## conv2 layer ##
W_conv2 = weight_variable([1, 3, 32, 64])  # kernel 5*5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_1x2(h_conv2)  # output size 7*7*64
## funcl layer ##
W_fc1 = weight_variable([1 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples,7,7,64]->>[n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 1 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## func2 layer ##
W_fc2 = weight_variable([1024, 15])
b_fc2 = bias_variable([15])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='softmax')
## acc ##
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)
# #################优化神经网络##################################
# the error between prediction and real data
pre_max = tf.math.reduce_max(prediction, name="pre_max")
clip = tf.clip_by_value(prediction,1e-25,500)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(clip), reduction_indices=[1]))  # loss
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
sess = tf.Session()
# important step

# ##################Tensorboar_Summary############
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(r"C:\tflog\train", sess.graph)
test_writer = tf.summary.FileWriter(r"C:\tflog\test")
sess.run(tf.initialize_all_variables())
#writer = tf.summary.FileWriter(r"C:\tflog", sess.graph)
# console  C:\Users\70951\AppData\Roaming\Python\Python37\Scripts\tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"
# #################优化神经网络##################################


print(dataset.test.images.shape)
#xs_test = dataset.test.images[0:10000, :, :].reshape((-1, 8*window_size))
#ys_test = dataset.test.labels[0:10000, :].reshape((-1, 15))
batch = dataset.test.next_batch(2000)
xs_test = batch[0].reshape(-1, 8 * 32)
ys_test = batch[1]
acc_test = numpy.zeros(2000)
acc_train = numpy.zeros(2000)
acci = 0
variable_summaries(xs_test)
#21907
for epoch in range(400):
    if epoch < 5000:
        ra = 1e-4
    elif epoch < 10000:
        ra = 5e-5
    elif epoch < 20000:
        ra = 1e-5
    else:
        ra = 5e-6
    batch = bear.train.next_batch(200)
    xs_train = batch[0].reshape(-1, 8 * 32)
    ys_train = batch[1]

    summary, _ = sess.run([merged, train_step], feed_dict={xs: xs_train, ys: ys_train, keep_prob: 0.5, rate: ra})
    #train_writer.add_summary(summary, epoch)
    if epoch % 100 == 0:
        # print(sess.run(prediction,feed_dict={xs: batch_xs}))
        #summary, output = sess.run([merged, 'softmax:0'], {xs: xs_test,keep_prob: 0.5})
        #assert not(numpy.isnan(output.any()))
        summary, acc_train[acci] = sess.run([merged, accuracy], {xs: xs_train, ys: ys_train, keep_prob: 1, rate: ra})
        train_writer.add_summary(summary, epoch)
        summary, acc_test[acci], y_pred = sess.run([merged, accuracy, prediction], {xs: xs_test, ys: ys_test, keep_prob: 1, rate: ra})
        #acc_test[acci] = compute_accuracy(xs_test,ys_test)
        #acc_train[acci] = compute_accuracy(xs_train, ys_train)
        acci = acci + 1
        print(epoch, " test_acc={:.4f}  train_acc={:.4f}  ".format(acc_test[acci-1], acc_train[acci-1]))
        test_writer.add_summary(summary, epoch)

# console  C:\Users\70951\AppData\Roaming\Python\Python37\Scripts\tensorboard --logdir train:"C:\tflog\train",test:"C:\tflog\test"



def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat)
    plt.title('Confusion Matrix')
    plt.colorbar()

    labels = ['None', 'E1', 'W1', 'E2', 'W2', 'E3', 'W3', 'E4', 'W4', 'E1W1', 'E2W2', 'E3W3', 'E4W4', 'E1E4', 'E2E3']
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == '__main__':
    print(y_pred.shape)
    y_true = numpy.zeros((500, 1))
    for i in range(500):
        y_true[i, 1] = numpy.argmax(y_pred[i, :])
    confusion_mat = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(confusion_mat)




