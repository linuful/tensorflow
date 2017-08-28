#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from datetime import datetime
from numpy import *
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def tensor():
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/tensorboard", sess.graph)
        sess.run(init)
        #with tf.device("/gpu:1"):
        with tf.device("/cpu:0"):
            for i in range(10000):
                matrix1 = tf.constant([[3., 3.]])
                matrix2 = tf.constant([[2.],[3.]])
                product = tf.matmul(matrix1, matrix2)
                #tf.nn.relu()
                sess.run(product)
                if i%50==0:
                    result = sess.run(merged)
                    writer.add_summary(result, i)
                    print(product.eval(),product.shape)


def show():
    x=0
    y=0;
    z=x+y;
    p=print(str(x)+","+str(y)+","+str(z))
    return x,y,p

if __name__ == '__main__':

    print (datetime.now())
    tensor()

