# 设置学习率
learning_rate = 0.01
# 设置训练次数
train_steps = 1000

import tensorflow as tf
import numpy
import matplotlib as plt
from old import input_data

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



mnist = input_data.read_data_sets(images, labels, test_images, test_labels, one_hot=False)
with tf.name_scope('data'):
    # 可修改批处理数
    batch = mnist.train.next_batch(100)
    x_data = batch[0].reshape(-1, (CYCLE + 1) * MEASURE * STATE)
    y_data = batch[1]

with tf.name_scope('Input'):
    # 256个节点
    input1 = tf.placeholder(tf.float32, [None, (CYCLE + 1) * MEASURE * STATE])
    weight1 = tf.Variable(tf.truncated_normal([(CYCLE + 1) * MEASURE * STATE, 256], stddev=0.005))
    bias1 = tf.Variable(tf.truncated_normal([256], stddev=0.005))
    output1 = tf.add(tf.matmul(input1, weight1), bias1)

with tf.name_scope('Layer1'):
    # 10个节点
    weight2 = tf.Variable(tf.ones([256, 20]))
    bias2 = tf.Variable(tf.ones([20]))
    output2 = tf.add(tf.matmul(output1, weight2), bias2)

with tf.name_scope('Layer2'):
    # 10个节点
    weight3 = tf.Variable(tf.ones([20, 10]))
    bias3 = tf.Variable(tf.ones([10]))
    output3 = tf.add(tf.matmul(output2, weight3), bias3)

with tf.name_scope('Layer3'):
    # 10个节点
    weight4 = tf.Variable(tf.ones([10, 10]))
    bias4 = tf.Variable(tf.ones([10]))
    output4 = tf.add(tf.matmul(output3, weight4), bias4)

with tf.name_scope('layer4'):
    # 10个节点
    weight5 = tf.Variable(tf.ones([10, 10]))
    bias5 = tf.Variable(tf.ones([10]))
    output5 = tf.add(tf.matmul(output4, weight5), bias5)

# 输出
with tf.name_scope('Prediction'):
    weight6 = tf.sigmoid(tf.Variable(tf.ones([10, 8])))
    bias6 = tf.Variable(tf.ones([8]))
    output6 = tf.add(tf.matmul(output5, weight6), bias6)
    Target = tf.placeholder(tf.float32, [None, 8])

# 采用交叉熵作为损失函数
with tf.name_scope('Loss'):
    sq = tf.square(output6 - Target)
    loss = tf.reduce_mean(sq)

# 定义训练操作
with tf.name_scope('Train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
with tf.name_scope('Init'):
    init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    l = []

    # 循环1000次
    for i in range(100000):
        sess.run(train_op, feed_dict={input1: x_data, Target: y_data})
        lo = sess.run(loss, feed_dict={input1: x_data, Target: y_data})
        print(lo)
        l.append(lo)
    print("Optimization Finished!")

    plt.plot(l)
    plt.xlabel('The sampling point')
    plt.ylabel('loss')
    plt.title("The variation of the loss")
    plt.grid(True)
    plt.show()
    # 写入日志文件，可自行指定路径
    writer = tf.summary.FileWriter("logs/", sess.graph)
