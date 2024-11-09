import numpy as np
from myNNetwork import FCN
from MnistProcessing import Train_images,Train_label,Test_images,Test_label
import pickle

input_size = 784
h1_size = 256
h2_size = 128
output_size = 10

Network = FCN(input_size, h1_size, h2_size, output_size,weight_init_std=0.01)
Network.reset_factor()

x_train = Train_images
t_train = Train_label
x_test  = Test_images
t_test  = Test_label

epoch = 16
batch_size = 100
t_data_num = 60000
lr = 0.2
item = 6000 #投入训练的样本总批次数
train_size = x_train.shape[0]     #60000
count = 0

train_loss_list = []             #储存训练损失值
train_acc_list = []              #储存训练正确值
test_acc_list = []               #储存测试正确值
test_loss_list = []              #储存测试损失值
test_count_list = []


for i in range(1,1+item):
    batch_maks = np.random.choice(train_size, batch_size)  # 在60000里随机抽取100个批次
    x_batch = x_train[batch_maks]
    t_batch = t_train[batch_maks]

    x_loss = Network.loss(x_batch,t_batch)#计算误差，内置了前向传播函数

    train_loss_list.append(x_loss)

    grad = Network.gradient()

    for key in ('W1', 'b1', 'W2', 'b2' , 'W3' , 'b3'):
        Network.factor[key] -= lr * grad[key]

    if i % 100 == 0:  # N个迭代记录一次

        count = count +1

       #train_acc = Network.accuracy(x_train, t_train)
        test_acc = Network.accuracy(x_test, t_test)

        #train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        test_loss_list.append("%.5f" % x_loss)
        test_count_list.append(count)

        acc = 100 * float(test_acc_list[-1])

        print(f'当前进度：{i}/{item}   正确率：{acc:.2f}%   loss值：{test_loss_list[-1]}')

import pandas as pd

def save_metrics_csv(count_list, loss_list, acc_list, file_path):
    data = {'count': count_list, 'Loss': loss_list, 'Accuracy': acc_list}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

accuracy = 100*float(test_acc_list[-1])
save_path = r"E:\pycharm\mypython\pythonProject\model\model_acc_{acc:.2f}.pkl".format(acc=accuracy)
with open(save_path,'wb') as f:
    pickle.dump(Network,f)

save_data_path = r"E:\pycharm\mypython\pythonProject\model\model_acc_{acc:.2f}.metrics.csv".format(acc=accuracy)
save_metrics_csv(test_count_list, test_loss_list, test_acc_list, save_data_path)


print(f"最终正确率:{accuracy:.2f}%")



"""
df1 = pd.read_csv(r"E:\pycharm\mypython\pythonProject\model\model_acc_88.29.metrics.csv")
df2 = pd.read_csv(r"E:\pycharm\mypython\pythonProject\model\model_acc_88.23.metrics.csv")

loss_list1 = df1["Loss"].tolist()
acc_list1 = df1["Accuracy"].tolist()
loss_list2 = df2["Loss"].tolist()
acc_list2 = df2["Accuracy"].tolist()



import matplotlib.pyplot as plt

epochs = np.arange(len(loss_list1))  # 假设训练轮次与第一个模型数据长度相同

plt.plot(epochs, loss_list1, label="Model 1 Loss", marker='o')
plt.plot(epochs, loss_list2, label="Model 2 Loss", marker='s')
plt.plot(epochs, acc_list1, label="Model 1 Accuracy", marker='^', linestyle='--')
plt.plot(epochs, acc_list2, label="Model 2 Accuracy", marker='v', linestyle='-.')

plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Loss and Accuracy Comparison of Multiple Models")
plt.legend()
plt.show()
"""