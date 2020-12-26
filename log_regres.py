import numpy as np
from numpy import*
import matplotlib.pyplot as plt

def load_dataset():
    #初始化数据向量和标签向量
    data_matrix = []
    label_matrix = []

    #加载文件
    file_name = open('testSet.txt','rb')

    #遍历文件
    for line in file_name.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0,float(line_array[0]),float(line_array[1])])
        label_matrix.append(int(line_array[2]))

    return data_matrix, label_matrix

#sigmoid()函数
def sigmoid(x):
    return 1 / (1 + exp(-x))

#梯度上升算法-解回归系数
def grandient_ascent(input_date,class_lables):
    data_matrix = mat(input_date)#将输入列表变为np矩阵化
    label_matrix = mat(class_lables).transpose()#将输入列表np矩阵化后，转置。transpose()函数，这里1*100变为100*1
    #获取矩阵的形状，这里为100*3的矩阵
    m,n = shape(data_matrix)
    #设置步长
    alpha = 0.001
    #设置迭代次数
    max_cycles = 500
    #初始化回归系数，3*1的一个1矩阵
    weights = ones((n,1))

    for k in list(range(max_cycles)):
        h = sigmoid(data_matrix * weights) 
        error = label_matrix - h #计算梯度
        weights = weights + alpha * data_matrix.transpose()*error #计算回归系数

    return weights

#随机梯度上升
def random_gradient_ascent(data_matrix,class_lables):
    m,n = shape(data_matrix)#m是行，n是列
    #初始化步长和回归因子
    alpha = 0.01
    weights = ones(n)#n个1的一维数组

    for i in range(m):
        h = sigmoid(sum(data_matrix[i]*weights))
        error = class_lables[i] - h
        weights = weights + alpha*error*data_matrix[i]

    return weights

#改进的随机梯度上升
def random_gradient_ascent1(data_matrix,class_lables,num_iter = 150):#mo默认迭代次数150次

    m,n = shape(data_matrix)#m是行，n是列
    weights = ones(n)#n个1的一维数组
    #迭代过程
    for j in list(range(num_iter)):
        #数据的索引
        data_index = list(range(m))
        #遍历数据点
        for i in list(range(m)):
            alpha = 4 / (1.0 + j + i) +0.01 #步长随迭代次数增加不断减少，但最小为后面的常数  ？？？怎么来的，为什么这样选
            
            #在0到len(data_index)范围内随机选取一个数，并删除原列表中的值
            rand_index = int(random.uniform(0,len(data_index)))#通过随机选取样本更新回归系数，从而减少周期性的波动

            h = sigmoid(sum(data_matrix[rand_index]*weights))
            error = class_lables[rand_index] - h
            weights = weights + alpha*error * data_matrix[rand_index]
            del(data_index[rand_index])
        return weights

#画出数据集和logistic回归最佳拟合直线
def best_fit(weights):
    #加载数据向量和标签向量
    data_matrix, label_matrix = load_dataset()
    #将数据向量转为np数组
    data_array = np.array(data_matrix)
    #获取数组的维数即数据点的个数，[0]表示第二维-行数，[1]表示第一维-列数
    n = shape(data_array)[0]

    x_cord1 = [];y_cord1 = []
    x_cord2 = [];y_cord2 = []
    #遍历数据点
    for i in list(range(n)):
        #如果是其标签为1，则将其保存在(x_cord1,y_cord1)
        if int(label_matrix[i]) == 1:
            x_cord1.append(data_array[i,1])
            y_cord1.append(data_array[i,2])
        #否则将其保存在（x_cord2,y_cord2)
        else:
            x_cord2.append(data_array[i,1])
            y_cord2.append(data_array[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)#添加子图
    #用散点画出数据点，标签为1的为蓝色，标签为0的为红色
    ax.scatter(x_cord1,y_cord1,s=30,c='blue',marker = 's')
    ax.scatter(x_cord2,y_cord2,s=30,c='red')
    #给点x,y的范围，其中0=w0*x0+w1*x1+w2*x2,所有X2变换为下面公式
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('x2')
    plt.show()


#使用logistic回归进行分类
#思路，将测试集中的每个特征向量乘以最优化方法得到的回归系数，然后再求和代如sigmoid()函数里，大于0.5归为1
def classify_vector(inX,weights): #inX表示特征向量
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0 

def colic_test():
    #打开训练集和测试集
    fr_train = open('horseColicTraining.txt', 'rb')
    fr_test = open('horseColicTest.txt', 'rb')

    #初始化训练数据集和标签
    train_set = []; train_labels = []
    #读取训练集中的数据并用制表符分隔开来
    for line in fr_train.readlines():

        current_line = line.decode().strip().split('\t')
        #初始化数组
        line_array = []
        #一共22列，其中最后一列为标签，读取每一行后将每一行的数据都加载到train_set中，标签放到train_labels里
        for i in list(range(21)):
            line_array.append(float(current_line[i]))
        train_set.append(line_array)
        train_labels.append(float(current_line[21]))
    #回归系数的计算，迭代500次
    train_weights = random_gradient_ascent1(array(train_set),train_labels,500)

    error_count = 0; num_testvec = 0
    #读取测试数据并处理
    for line in fr_test.readlines():
        num_testvec += 1.0
        #python3.在解码上和python2.有所区别decode()将bytes变为str，而encode()则将str变为bytes
        current_line = line.decode().strip().split('\t')
        line_array = []
        for i in list(range(21)):
            line_array.append(float(current_line[i]))

        #判断训练算法预测出来的结果是否符合，如果不符合错误次数加1
        if int(classify_vector(array(line_array),train_weights)) != int(current_line[21]):  
            error_count += 1
    #错误率等于错误次数比总的测试次数
    error_rate = (float(error_count)/num_testvec)
    print('The error rate of this test is: %f' % error_rate)
    return error_rate

#多次测试取平均值
def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in list(range(num_tests)):
        error_sum += colic_test()
    print("After %d iterations the average error rate is:%f "%(num_tests, error_sum/float(num_tests)))

