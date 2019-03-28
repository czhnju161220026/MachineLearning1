#coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数


def fun(x, y):
    return x * x + y * y  + 3 * x + 4 * y


class Finder:
    def __init__(self):
        self.step = 0.01
        self.break_condition = 0.01
        self.beta = np.array([-15, 10])
        self.points_X = []
        self.points_Y = []

    def calculate_gradient(self):
        result = np.array([0,0])
        result[0] = 2 * self.beta[0] + 3
        result[1] = 2 * self.beta[1] + 4
        return result

    def find(self):
        a1 = fun(self.beta[0] , self.beta[1])
        while True:
            gradient = self.calculate_gradient()
            self.beta = self.beta - self.step * gradient
            a2 = fun(self.beta[0] , self.beta[1])
            if abs(a2 - a1) < self.break_condition:
                break
            else:
                self.points_X.append(self.beta[0])
                self.points_Y.append(self.beta[1])
                a1 = a2
    def getPoints(self):
        return self.points_X,self.points_Y


fig1 = plt.figure()  # 创建一个绘图对象
ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
X = np.arange(-15, 15, 0.1)
Y = np.arange(-15, 15, 0.1)  # 创建了从-2到2，步长为0.1的arange对象
# 至此X,Y分别表示了取样点的横纵坐标的可能取值
# 用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点
X, Y = np.meshgrid(X, Y)
Z = fun(X, Y)  # 用取样点横纵坐标去求取样点Z坐标
plt.title("This is main title")  # 总标题
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
ax.set_xlabel('x label', color='r')
ax.set_ylabel('y label', color='g')
ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
finder = Finder()
finder.find()
X1,Y1 = finder.getPoints()
Z1 = []
for i in range(0, len(X1)):
    Z1.append(fun(X1[i], Y1[i]))
ax.scatter(X1,Y1,Z1,c='r')

plt.show()  # 显示模块中的所有绘图对象

a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a,b))
a.reshape((3,1))
a= np.mat(a)
b = np.mat(b)
print(a)
print(np.multiply(a.transpose(),b))
