import numpy as np
import os
import struct



# 声明一些变量存储数据维度等信息
train_num = 60000
test_num = 10000
img_size = 784


# 保存训练集和测试集文件的路径
f_path = r'E:\pycharm\mypython\pythonProject：End\data\MNIST\raw'


"""
    从指定路径加载 MNIST 数据集。
    :param path: 数据集路径。
    :param kind: 'train' 或 't10k'，表示训练集或测试集。
    :return: 图像数据和标签。
    :OneHot:是否将正确标签one_hot化
    """

def load_mnist(path,kind='',onehot=False):

    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)


    if onehot:
        T = np.zeros((labels.size,10)) #创建一个和标签数组一样的全0数组
        for i,r in enumerate(T):
            r[labels[i]] = 1           #遍历数组并把对应的标签定为1

        return T

    else:
       return images / 255.0 #归一化

Train_images = load_mnist(f_path,kind='train',onehot=False)
Test_images  = load_mnist(f_path,kind='t10k',onehot=False)
Train_label  = load_mnist(f_path,kind='train',onehot=True)
Test_label   = load_mnist(f_path,kind='t10k',onehot=True)

#print(Train_images.shape)
#print(Test_images.shape)
#print(Train_label.shape)
#print(Test_label.shape)
