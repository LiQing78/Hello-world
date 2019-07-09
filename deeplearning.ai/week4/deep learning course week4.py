import numpy as np
import h5py
import matplotlib.pyplot as plt

# pylot使用rc配置文件来自定义图形的各种默认属性，称之为rc配置或rc参数。
# 图像显示大小
plt.rcParams["figure.figsize"] = (5.0, 4.0)
# 最近邻差值: 像素为正方形
# Interpolation/resampling即插值，是一种图像处理方法，
# 它可以为数码图像增加或减少象素的数目。
plt.rcParams['image.interpolation'] = 'nearest'
# 使用灰度输出而不是彩色输出
plt.rcParams['image.cmap'] = 'gray'

arr3D = np.array([[[1, 1, 2, 2,3, 4],
                  [1, 1, 2, 2, 3, 4],
                  [1, 1, 2, 2, 3, 4]],
                 [[0, 1, 2, 3, 4, 5],
                  [0, 1, 2, 3, 4, 5],
                  [0, 1, 2, 3, 4, 5]],
                 [[1, 1, 2, 2, 3, 4],
                  [1, 1, 2, 2, 3, 4],
                  [1, 1, 2, 2, 3, 4]]])
# np.pad(array, pad_width, mode='constant)
# pad_width -- 表示每个轴（axis）边缘需要填充的数值数目
# (0,0)表示在z维度padding = 0
# (1，1)表示在水平维度方向上、下 padding = 1
# (2, 2)表示在垂直维度方向左、右 padding = 2
# mode -- 表示填充方式“constant表示连续填充相同的值"
print("contant:\n" + str(np.pad(arr3D, ((0, 0), (1, 1), (2, 2)), 'constant')))

def zero_pad(X, pad):
    """
    把数据集X的图像全部使用0来扩充pad个宽度和高度
    :param X: 图像数据集，维度为（样本数，图像高度，图像宽带，图像通道数）
    :param pad: 整数，每个图像在垂直和水平维度上的填充量
    :return: X_paded - 扩充后的图像数据集，维度为（样本数，图像高度+2*pad，图像宽度+2*pad, 图像通道数）
    """
    X_paded = np.pad(X, ((0,0), (pad,pad), (pad, pad), (0,0)),
                     "constant", constant_values=0)
    return X_paded

