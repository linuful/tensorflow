#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PIL import Image
import struct
import gzip
import os

MNIST_DATA_PATH='MNIST_data'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


def read_image(filename, path = MNIST_DATA_PATH+"/picture/"):
    print("saving "+filename)
    os.makedirs(path, exist_ok=True)
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')#使用大端规则
    for i in range(images):
        image = Image.new('L', (columns, rows))
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')#读取一个字节
        if(i%1000==0):
            print('save ' + str(i) + '.png')
        image.save(path +"/"+ str(i) + '.png')
    print("save "+filename+" success!")


def read_label(filename, saveFilename):
    print("saving " + filename)
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labelArr = [0] * labels
    for x in range(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
    save = open(saveFilename, 'w+')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print ('save labels success')

def un_gz(file_name):
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()

def extract_mnist():
    un_gz(MNIST_DATA_PATH + "/" + TRAIN_IMAGES)
    un_gz(MNIST_DATA_PATH + "/" + TRAIN_LABELS)
    un_gz(MNIST_DATA_PATH + "/" + TEST_IMAGES)
    un_gz(MNIST_DATA_PATH + "/" + TEST_LABELS)
    print("extract success!")



if __name__ == '__main__':
    filename="MNIST_data/train-images-idx3-ubyte"
    if(False==os.path.exists(filename)):
        extract_mnist()
    read_image(MNIST_DATA_PATH +"/"+TRAIN_IMAGES.replace(".gz", ""),path=MNIST_DATA_PATH+"/picture/train")
    read_image(MNIST_DATA_PATH + "/" + TEST_IMAGES.replace(".gz", ""),path=MNIST_DATA_PATH+"/picture/test/")
    read_label(MNIST_DATA_PATH + "/" + TRAIN_LABELS.replace(".gz", ""),MNIST_DATA_PATH+"/picture/train_label.txt")
    read_label(MNIST_DATA_PATH + "/" + TEST_LABELS.replace(".gz", ""), MNIST_DATA_PATH + "/picture/test_label.txt")