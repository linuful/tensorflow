import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from numpy import *
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import tensorflow as tf

import mnist_image

#创建权重
def weight_variable(shape):
    # 标准差0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#创建偏置
def bias_variable(shape):
    #偏置0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#2维卷积层
#W 卷积参数 [5,5,1,32]:[卷积核的shape,图片的RGB channel，提取的特征数量]
#strides=[1, 1, 1, 1]: 移动步长
#padding:边界处理方式，SAME表示给边界加padding，让卷积的输入输出保持相同大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#最大池化（2*2-1*1）
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def build_CNN():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])#输入
    y_ = tf.placeholder(tf.float32, [None, 10])#真实的输出
    x_image = tf.reshape(x, [-1, 28, 28, 1]) #将1*784 -> 28*28,-1代表样本数不固定

    #第一个卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])#5*5的卷积核，1个颜色通道32个不同的卷积核(特征值)
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #第二个卷积层 64个卷积核（特征）
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  #第2层卷积之后，经过2次池化图片变成7*7，卷积核数64，则输出tensor shape为7*7*64

    #全连接层
    #设置用1024个隐含感知神经元节点，1M个神经元
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) #变成1维矩阵
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Dopout层，随机丢弃一部分节点数据减轻过拟合
    #预测是保留全部数据以达到好的效果
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #softmax层，得出概率输出
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    #交叉熵损失函数，使用adm优化器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.global_variables_initializer().run()

    for i in range(100):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            #验证准确性时keep_prob设置为1
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    predicted=tf.cast(tf.argmax(y_conv, 1),tf.int32)
    return x,y_,predicted,keep_prob
if __name__ == '__main__':
    sess = tf.InteractiveSession()
    x, y, predicted,keep_prob=build_CNN()
    try:
        test_images1, test_labels1 = mnist_image.GetImage("D:/2.png")
        print("input:", tf.arg_max(test_labels1, 1).eval()[0])
        test = DataSet(test_images1, test_labels1, dtype=tf.float32)

        res=predicted.eval({x: test.images, y: test.labels,keep_prob: 1})
        print("output:",res[0])
    except BaseException as e:
        print("exception:",e)
    finally:
        pass

    dir_name="images"
    files = os.listdir(dir_name)
    cnt=len(files)
    for i in range(cnt):
        files[i]=dir_name+"/"+files[i]
        test_images1,test_labels1=mnist_image.GetImage(files[i])
        print("input:",tf.arg_max(test_labels1,1).eval()[0])
        test = DataSet(array(test_images1), array(test_labels1), dtype=tf.float32)
        res=predicted.eval({x: test.images, y: test.labels,keep_prob:1.0})
        print("output:", res[0])
        print("\n")

    while(True):
        print("Please input a image file:")
        filename=input();
        try:
            test_images1, test_labels1 = mnist_image.GetImage(filename)
            #print("input:", tf.arg_max(test_labels1, 1).eval()[0])
            print("input file:%s,number:%d"%(filename,tf.arg_max(test_labels1, 1).eval()[0]))
            test = DataSet(test_images1, test_labels1, dtype=tf.float32)

            res = predicted.eval({x: test.images, y: test.labels, keep_prob: 1})
            print("output:", res[0])
        except BaseException as e:
            print(e)
        finally:
            pass



