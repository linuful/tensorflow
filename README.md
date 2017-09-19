# tensorflow
用于管理深度学习神经网络相关学习的笔记代码,该project是基于TensorFlow example的二次开发

数据：

  Mnist_data是官方的mnist数据库
  
  images 中存放待预测的图片
  
以下代码基于TensorFlow1.2 Python 3.5

  demo.py 为调试函数用的文件
  
  fully_connected_feed.py 为全连接模型
  
  mnist_with_summaries.py 为tensorboard相关演示，tensorboard启动方式：tensorboard --logdir=./logs
  
  mnist_cnn.py MNIST的CNN模型实现
  
  mnist_d2pic.py 将MNIST数据集还原成图片
  
  mnist_image.py 将图片转换成矩阵，目前只能处理28*28灰度图片，图片的label通过图片名首字符获得
  
  mnist_softmax.py tensorflow官方examples\tutorials
  
  mnist_softmax_me.py 本项目的softmax模型
  
  sigmoid.py 实现sigmoid函数
  
  state_update.py 状态机演示
  


//code operator

1. source activate tensorflow

2. pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl

3. pyton demo.py

4. tensorboard --logdir=/tmp/tensorflow/mnist/logs/mnist_with_summaries

5. source deactivate

