#coding=utf-8
from Preprocess import Sample, TrainSet, split, get_test_samples
# 第三方库
import numpy as np


# TODO: 一个二分类器
class BinaryClassfier:
    def __init__(self, beta):
        self.beta = beta
        self.precision = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def display_parameter(self):
        print(self.beta)

    def set_confusion_matrix(self, confusion_matrix):
        self.tp = confusion_matrix[0]
        self.fp = confusion_matrix[1]
        self.tn = confusion_matrix[2]
        self.fn = confusion_matrix[3]

    def get_accuracy(self):
        return (self.tn + self.tp) / (self.tp + self.fn + self.tn + self.fp)

    # 返回对数概率
    def classify(self, sample):
        x_hat = np.array(sample.get_features())
        logistic_probability = np.dot(self.beta, x_hat)
        return logistic_probability

    def get_confusion_matrix(self):
        return [self.tp, self.fp, self.tn, self.fn]

    def export_model(self, fout):
        for i in self.beta:
            fout.write(str(i) + ' ')
        fout.write('\n')


# 训练单个模型
class SingleTrainer:
    def __init__(self, train_set):
        self.train_set = train_set
        self.classifiers = []
        self.m = train_set.size()
        self.dimension = train_set.get_dimension()
        self.beta = np.array([0 for i in range(self.dimension)])

    # 对数似然函数(已经转化为要求最小化的形式)
    def log_likelihood_function(self):
        result = 0.0
        for i in range(0, self.m):
            sample = self.train_set.get_sample(i)
            y_i = sample.get_label()
            x_hat = np.array(sample.get_features())
            result = result + (-1.0 * y_i * np.dot(self.beta, x_hat) + np.log(
                1 + np.power(np.e, np.dot(self.beta, x_hat))))
        return result

    # 计算一阶导数(梯度)
    def calc_gradient(self):
        result = np.array([0.0 for i in range(self.dimension)])
        for i in range(0, self.m):
            sample_i = self.train_set.get_sample(i)
            y_i = sample_i.get_label()
            x_hat = np.array(sample_i.get_features())
            result = result - (x_hat * (y_i - (1.0 - 1.0 / (1.0 + np.power(np.e, np.dot(self.beta, x_hat))))))
        return result

    # 计算二阶导数
    def calc_second_derivative(self):
        result = np.mat(np.zeros((self.dimension, self.dimension)))
        for i in range(0, self.m):
            sample_i = self.train_set.get_sample(i)
            x_hat = np.array(sample_i.get_features())
            p_1 = 1.0 - 1.0 / (1.0 + np.power(np.e, np.dot(self.beta, x_hat)))
            result = result + (np.matmul(np.mat(x_hat).transpose(), np.mat(x_hat)) * p_1 * (1 - p_1))
        return result

    def display(self):
        print("Size:" + str(self.m) + " ,dimension: " + str(self.dimension))
        print("Beta: " + str(self.beta))
        return

    # 梯度下降法 : 收敛实在太慢
    def train_gradient_descent(self):
        # TODO:先尝试固定步长，再尝试迭代过程中更新步长
        step = 0.00001
        break_condition = 0.00001
        a1 = self.log_likelihood_function()
        count = 1
        while True:
            gradient = self.calc_gradient()
            self.beta = self.beta - step * gradient
            # print("Update beta: "+str(self.beta))
            a2 = self.log_likelihood_function()
            # print("log_likelihoot: "+ str(a2))
            if abs(a2 - a1) < break_condition:
                break
            else:
                if a2 > a1:
                    if step > 0.0000025:
                        step = step * 0.9
                a1 = a2
                if count % 1000 == 0:
                    print("l(beta) = " + str(a2))
            count = count + 1
            if count == 20000:
                break
        print("After train: beta:" + str(self.beta))
        print("Likelihood = " + str(a2))
        print("count:" + str(count))

    # 牛顿法
    def train_newton_method(self):
        print("Newton Method")
        break_condition = 0.00001
        a1 = self.log_likelihood_function()
        count = 1
        while True:
            second_derivative = self.calc_second_derivative()
            gradient = self.calc_gradient()
            gradient = np.mat(gradient).transpose()
            delta = np.matmul(second_derivative.I, gradient).transpose()

            self.beta = self.beta - np.array(delta.tolist()[0])
            a2 = self.log_likelihood_function()
            print("log_likelihood: " + str(a2))
            if abs(a2 - a1) < break_condition:
                break
            a1 = a2
            if count % 1000 == 0:
                print("l(beta) = " + str(a2))
            count = count + 1
            if count == 20000:
                break

        print("After train: beta:" + str(self.beta))

    def get_classifier(self):
        return BinaryClassfier(self.beta)


# 训练多个模型
class MultiTrainer:
    def __init__(self, groups):
        self.groups = groups
        self.classifiers = []

    def generate_classifiers(self):
        for i in range(26):
            print("-------------------Training Classifier: %s---------------" % str(i))
            positive_set = self.groups[i]
            negative_set = split(i + 1, positive_set.size() * 15, self.groups)
            train_set = TrainSet(positive_set, negative_set)
            trainer = SingleTrainer(train_set)
            # trainer.train_gradient_descent() 使用牛顿法而非梯度下降法
            trainer.train_newton_method()
            classifier = trainer.get_classifier()
            self.classifiers.append(classifier)
        fout = open("models.txt", 'w')
        for classifier in self.classifiers:
            classifier.export_model(fout)
        fout.close()

    def get_classifiers(self):
        return self.classifiers
