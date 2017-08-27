# tensorflow
用于管理深度学习神经网络相关代码
数据：
  Mnist_data是官方的mnist数据库
  images 中存放待预测的图片，目前可以放28*28的灰度图或者rgb图，图片的命名格式为 标准数字_序号

以下代码基于TensorFlow1.2 Python 3.5
  demo.py 为调试函数用的文件
  fully_connected_feed.py 为全连接模型
  mnist_cnn.py MNIST的CNN模型实现
  mnist_d2pic.py 将MNIST数据集还原成图片
  mnist_image.py 将图片转换成矩阵，目前只接受28*28灰度图片
  mnist_softmax.py tensorflow官方examples\tutorials
  mnist_softmax_me.py 本项目的softmax模型
  sigmoid.py 实现sigmoid函数
  state_update.py 状态机演示


