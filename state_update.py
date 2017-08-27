#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
#export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
state=tf.Variable(0,name="counter")
one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init_op=tf.global_variables_initializer()
sess=tf.InteractiveSession()

with tf.device("/cpu:0"):
    init_op.run()
    print(state.eval())

    for _ in range(3):
        update.op.run()
        print(state.eval())

