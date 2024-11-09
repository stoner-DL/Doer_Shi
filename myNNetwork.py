import numpy as np
from collections import OrderedDict
import calculate



class FCN:
    def __init__(self,input_size, h1_size, h2_size , output_size, weight_init_std=0.01):
        self.weight_init_std = weight_init_std
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.output_size = output_size

        self.factor = {}
        self.layers = OrderedDict()
        self.lastLayer = None

    def reset_factor(self):
        self.factor["W1"] = self.weight_init_std * np.random.randn(self.input_size, self.h1_size)
        self.factor["b1"] = np.random.uniform(-0.1, 0.1, size=self.h1_size)
        self.factor["W2"] = self.weight_init_std * np.random.randn(self.h1_size, self.h2_size)
        self.factor["b2"] = np.random.uniform(-0.1, 0.1, size=self.h2_size)
        self.factor["W3"] = self.weight_init_std * np.random.randn(self.h2_size, self.output_size)
        self.factor["b3"] = np.random.uniform(-0.1, 0.1, size=self.output_size)

        self.layers["FCL1"] = calculate.FCL(self.factor["W1"], self.factor["b1"])
        self.layers["Relu1"] = calculate.Relu()
        self.layers["FCL2"] = calculate.FCL(self.factor["W2"], self.factor["b2"])
        self.layers["Relu2"] = calculate.Relu()
        self.layers["FCL3"] = calculate.FCL(self.factor["W3"], self.factor["b3"])
        self.lastLayer = calculate.SoftmaxWithLoss()


    def predict(self, x):
        """前向传播推理(不包括输出层 )"""
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """计算损失函数值-x:输入数据,t:监督数据"""
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """计算识别精度"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self):
        """通过反向传播法计算关于权重参数的梯度"""
        # 前向传播
        #self.loss(x, t)

        # 反向传播
        d_out = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)

        # 设定梯度字典，用于更新参数
        grads = {}
        grads["W1"], grads["b1"] = self.layers["FCL1"].dW, self.layers["FCL1"].db
        grads["W2"], grads["b2"] = self.layers["FCL2"].dW, self.layers["FCL2"].db
        grads["W3"], grads["b3"] = self.layers["FCL3"].dW, self.layers["FCL3"].db

        return grads

    def predicted_result(self, x):
        # 输出预测结果数字的方法

        result_arr = self.predict(x)
        result_label = np.argmax(calculate.softmax(result_arr))
        num_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        result_num = num_map[result_label]

        return result_num