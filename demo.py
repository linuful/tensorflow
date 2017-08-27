#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from datetime import datetime
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def tensor():
    with tf.Session() as sess:
        #with tf.device("/gpu:1"):
        with tf.device("/cpu:0"):
            matrix1 = tf.constant([[3., 3.]])
            matrix2 = tf.constant([[2.],[2.]])
            product = tf.matmul(matrix1, matrix2)
            tf.nn.relu()
            sess.run(product)
            print(product.eval(),product.shape)


def show():
    x=0
    y=0;
    z=x+y;
    p=print(str(x)+","+str(y)+","+str(z))
    return x,y,p

if __name__ == '__main__':
    print (datetime.now())

