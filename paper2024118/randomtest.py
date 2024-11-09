###用于测试模型实际准确率
import os.path
import  numpy as np
import pickle
from MnistProcessing import Test_images,Test_label

load_path = r'E:\pycharm\mypython\pythonProject\model\model_acc_97.76.pkl'
img_test   = Test_images
label_test = Test_label
right_count = 0

example_num = 5000            #测试用例数量
test_size = img_test.shape[0]

program0=0

for i in range(example_num):
    test_ind = np.random.choice(test_size, 1)
    img = img_test[test_ind]
    label = label_test[test_ind]
    Img =img.reshape(1,784)

    program =int( (i+1)/example_num*100 )

    if program!=program0:

        bar = f'[{"#" * (program * 50 // 100)}{"-" * (50 - (program * 50 // 100))}]{program}%'

        if program != 100:
            print(f'\r{bar}', end=' ')
        else:
            print(f'\r{bar}')
        program0 = program

        if program==100:
            print("done")

    with open(load_path,'rb') as f:
        loaded_model = pickle.load(f)

    result = loaded_model.predicted_result(Img)
    right_num = np.argmax(label)

    if result==right_num:
        right_count +=1

    img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
    img = img*255

file_name = os.path.basename(load_path)
print(f'测试模型:{file_name}')
print(f'正确个数：{right_count} 测试正确率：{100*right_count*1.0/example_num}%')