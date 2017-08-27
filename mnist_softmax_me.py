# coding=utf-8
# Import data
import os
from numpy import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import tensorflow as tf
import mnist_image

def build_NN():
    #import input
    #mnist = input.read_data_sets("data/", one_hot=True)
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #print(mnist.train.images.shape, mnist.train.labels.shape)
    #print(mnist.test.images.shape, mnist.test.labels.shape)
    #print(mnist.validation.images.shape, mnist.validation.labels.shape)

    sess = tf.InteractiveSession()
    # Create softmax  model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#使用梯度下降优化器，学习速率0,01

    # Train
    tf.global_variables_initializer().run()
    for i in range(1000):
      batch_xs, batch_ys = mnist_data.train.next_batch(100) #每次训练100个数据
      train_step.run({x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print( "accuracy:",sess.run(accuracy, feed_dict={x: mnist_data.test.images,y_: mnist_data.test.labels}))

    predicted=tf.cast(tf.argmax(y, 1),tf.int32)
    return x,y_,predicted

if __name__ == '__main__':
    x,y,predicted=build_NN();
    try:
        test_images1, test_labels1 = mnist_image.GetImage("D:/2.png")
        print("input:", tf.arg_max(test_labels1, 1).eval()[0])
        test = DataSet(test_images1, test_labels1, dtype=tf.float32)

        res=predicted.eval({x: test.images, y: test.labels})
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
        res=predicted.eval({x: test.images, y: test.labels})
        print("output:", res[0])
        print("\n")

