import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils

# 指定随机种子
np.random.seed(1)

# 初始化参数，对于一个两层的神经网络而言，
# 模型结构是线性-->ReLU-->线性-->sigmoid函数
def initialize_parameters(n_x, n_h, n_y):
    """
    初始化两层网络参数
    :param n_x: 输入层节点个数
    :param n_h: 隐藏层节点个数
    :param n_y: 输出层节点数量
    :return: 字典parameters,包含
            W1-权重矩阵，维度为(n_h,n_x)
            b1-偏差向量，维度为(n_h,1)
            W2-权重矩阵，维度为(n_y,n_h)
            b2-偏差向量，维度为(n_y,1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    # python的二维数据表示要用二层括号来进行表示
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 使用assert确保数据格式正确
    assert(W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

# print("==============测试initialize_parameters==============")
# parameters = initialize_parameters(3,2,1)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def initialize_parameters_deep(layers_dims):
    """
    初始化多层神经网络
    :param layers_dims: 网络中每个层的节点数量的列表
    :return: 字典parameters：W1，b1, W2, b2， ……， Wl, bl
            W1-权重矩阵，维度为(layers_dims[1], layers_dims[1-1])
            b1-偏差矩阵，维度为(layers_dims[1], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

        # 确保数据格式正确
        assert (parameters["W" + str(i)].shape == (layers_dims[i], layers_dims[i-1]))
        assert (parameters["b" + str(i)].shape == (layers_dims[i], 1))
    return parameters

# print("==============测试initialize_parameters_deep==============")
# layers_dims = [5,4,3]
# parameters = initialize_parameters_deep(layers_dims)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# 前向传播函数-->线性部分
def linear_forward(A, W, b):
    """
    实现前向传播的线性部分
    :param A: 来自上一层激活值，维度为(上一层节点数，样本数)
    :param W: 权重矩阵，维度为(当前层的节点数量，前一层的节点数量)
    :param b: 偏置向量，维度为(当前层的节点数量，1)
    :return: Z - 线性函数输出，激活函数的输入
            cache - 一个含有"A","W","b"的字典，用于后续的反向传播
    """
    Z = np.dot(W,A) + b
    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = {"W": W,
             "b": b,
             "A": A}
    return Z, cache

# print("==============测试linear_forward==============")
# A,W,b = testCases.linear_forward_test_case()
# Z,linear_cache = linear_forward(A,W,b)
# print("Z = " + str(Z))
# print("cache= " + str(linear_cache))

# 前向传播函数-->线性-激活部分
def linear_activation_forward(A_prev, W, b, activation):
    """
    实现线性-激活的前向传播
    :param A_prev: 来自上一层的激活，维度(上一层的节点数量，样本数)
    :param W: 权重矩阵，维度为(当前层的节点数量，前一层的节点数量)
    :param b: 偏置向量，维度为(当前层的节点数量，1)
    :param activation: 此层应用的激活函数名，relu - sigmoid
    :return: A - 激活函数输出值
            cache - 含有linear_cache, activation_cache的字典
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = {"linear_cache": linear_cache,
             "activation_cache": activation_cache}

    return A, cache

# print("==============测试linear_activation_forward==============")
# A_prev, W,b = testCases.linear_activation_forward_test_case()
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("sigmoid，A = " + str(A))
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("ReLU，A = " + str(A))
# print("cahce，A = " + str(linear_activation_cache))

def L_model_forward(X, parameters):
    """
    实现(linear->activation)*L-1  --> (linear->activation)层前向传播，
    :param X:输入数据，维度是(输入节点数，样本数)
    :param parameters:initialize_parameters_deep的输出
    :return:AL-最后一层的激活值
            caches - 缓存列表，包括：
            linear_relu_forward的cache，列表：L-1个，索引从0到L-2
            linear_sigmoid_forward的cache，列表：一个元素，索引为L-1
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(i)], parameters["b"+str(i)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

# print("==============测试L_model_forward==============")
# X,parameters = testCases.L_model_forward_test_case()
# AL,caches = L_model_forward(X,parameters)
# print("AL = " + str(AL))
# print("caches 的长度为 = " + str(len(caches)))
# print(caches)

# 计算成本函数 cost function
def compute_cost(AL, Y):
    """
    计算成本函数，
    :param AL: 预测向量，维度(1, 样本数)
    :param Y: 标签向量，维度(1, 样本数)
    :return: cost-成本
    """
    m = Y.shape[1]
    # np.multiply 对应元素相乘
    # np.dot 矩阵乘法
    cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), 1-Y)) / m
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost

# #测试compute_cost
# print("==============测试compute_cost==============")
# Y,AL = testCases.compute_cost_test_case()
# print("cost = " + str(compute_cost(AL, Y)))

# np.dot 和 np.multiply 的区别
# a1 = np.array([[1, 2],
#               [2, 5]])
# a2 = np.array([[1, 3],
#               [2, 9]])
# a3 = np.multiply(a1, a2)
# a4 = np.dot(a1, a2)
# print(a3, a4)

# 线性部分反向传播backward
def linear_backward(dZ, cache):
    """
    为某一层实现反向传播的线性部分(L层)
    :param dZ: 相对于当前层线性输出的成本梯度
    :param cache: 来自当前层前向传播的A_pre,W,b
    :return: dA_prev-相对于激活的成本梯度
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

# print("==============测试linear_backward==============")
# dZ, linear_cache = testCases.linear_backward_test_case()
#
# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

# 线性激活函数反向传播    linear->activation backward
def linear_activation_backward(dA, cache, activation="relu"):
    """
    实现linear->activation反向传播
    :param dA: 当前层激活后的梯度值
    :param cache: 用于计算反向传播的值,(linear_cache, activation_cache)
    :param activation:此层中使用的激活函数名称,(sigmoid, relu)
    :return:dA_prev-相对于前一层激活值的成本梯度值(前一层L-1)
            dW-相对于当前层W的成本梯度值(当前层L)
            db-相对于当前层b的成本梯度值(当前层L)
    """
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# print("==============测试linear_activation_backward==============")
# AL, linear_activation_cache = testCases.linear_activation_backward_test_case()
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")
#
# dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

# 构建多层模型反向传播函数
def L_model_backward(AL, Y, caches):
    """
    对(linear->relu)*L-1 --> (linear->sigmoid)执行反向传播(多层)
    :param AL: 正向传播的输出(L_model_forward)
    :param Y: 标签向量,维度是(1,样本数)
    :param caches: 包含linear_activation_forward(relu)--cache
                    包含linear_activation_forward(sigmoid)--cache
    :return: grads - 包含dA, dW, db 的字典
    """
    grads = {}
    L =len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)




