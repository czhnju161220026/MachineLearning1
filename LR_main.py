import numpy as np
import pandas as pd
import matplotlib


# 一个样本
class Sample:
    def __init__(self, features, label):
        self.features = features
        self.label = label

    def display(self):
        print("features: " + str(self.features) + " label: " + str(self.label))

    def get_features(self):
        return self.features

    def add_dimension(self):
        self.features.append(1.0)

    def get_dimension(self):
        return len(self.features)

    def get_label(self):
        return self.label


#  一类样本
class Group:
    def __init__(self, label):
        self.samples = []
        self.label = label

    def add_sample(self, sample):
        self.samples.append(sample)

    def size(self):
        return len(self.samples)

    def get_sample(self, index):
        return self.samples[index]


# 一组训练数据集
class TrainSet:
    def __init__(self, positive_set, negative_set):
        self.samples = []
        for i in range(0, positive_set.size()):
            sample_i = Sample(positive_set.get_sample(i).get_features(), 1)
            self.samples.append(sample_i)
        for i in range(0, negative_set.size()):
            sample_i = Sample(negative_set.get_sample(i).get_features(), 0)
            self.samples.append(sample_i)
        self.dimension = 0
        if len(self.samples) > 0:
            self.dimension = len(self.samples[0].get_features())

    def size(self):
        return len(self.samples)

    def get_sample(self, index):
        return self.samples[index]

    def get_dimension(self):
        return self.dimension

    def display(self):
        print("Num of samples: " + str(self.size()))
        print("Dimension: " + str(self.dimension))


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

    def export_model(self,fout):
        for i in self.beta:
            fout.write(str(i)+' ')
        fout.write('\n')



# 训练单个模型
class SingleTrainer:
    def __init__(self, train_set):
        self.train_set = train_set
        self.classifiers = []
        self.m = train_set.size()
        self.dimension = train_set.get_dimension()
        self.beta = np.array([0 for i in range(self.dimension)])

    def log_likelihood_function(self):
        result = 0.0
        for i in range(0, self.m):
            sample = self.train_set.get_sample(i)
            y_i = sample.get_label()
            x_hat = np.array(sample.get_features())
            result = result + (-1.0 * y_i * np.dot(self.beta, x_hat) + np.log(
                1 + np.power(np.e, np.dot(self.beta, x_hat))))
        return result

    def calc_gradient(self):
        result = np.array([0.0 for i in range(self.dimension)])
        for i in range(0, self.m):
            sample_i = self.train_set.get_sample(i)
            y_i = sample_i.get_label()
            x_hat = np.array(sample_i.get_features())
            result = result - (x_hat * (y_i - (1.0 - 1.0 / (1.0 + np.power(np.e, np.dot(self.beta, x_hat))))))
        return result

    def calc_second_derivative(self):
        result = np.mat(np.zeros((self.dimension, self.dimension)))
        for i in range(0, self.m):
            sample_i = self.train_set.get_sample(i)
            y_i = sample_i.get_label()
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
        print("Likelihood = " + str(a2))
        print("count:" + str(count))

    def get_classifier(self):
        return BinaryClassfier(self.beta)


# 训练多个模型
class MultiTrainer:
    def __init__(self, groups):
        self.groups = groups
        self.classifiers = []

    def generate_classifiers(self):
        for i in range(26):
            print("Training Classifier:" + str(i))
            positive_set = self.groups[i]

            negative_set = split(i + 1, positive_set.size() * 9.5, self.groups)
            train_set = TrainSet(positive_set, negative_set)
            trainer = SingleTrainer(train_set)
            # trainer.train_gradient_descent() 使用牛顿法而非梯度下降法
            trainer.train_newton_method()
            classifier = trainer.get_classifier()
            classifier.set_confusion_matrix(test_single_classifier(classifier, i + 1))
            self.classifiers.append(classifier)
        fout = open("models.txt",'w')
        for classifier in self.classifiers:
            classifier.export_model(fout)
        fout.close()

    def get_classifiers(self):
        return self.classifiers


# 测试者
class Tester:
    def __init__(self, _classifiers, _test_samples):
        self.classifiers = _classifiers
        self.test_samples = _test_samples

    def test(self):
        n = len(self.test_samples)
        right_case = 0
        wrong_case = 0
        classification_matrix = np.zeros((26, 26))
        for sample in self.test_samples:
            logits = []
            for classifier in self.classifiers:
                logits.append(classifier.classify(sample))
            result = np.argmax(np.array(logits)) + 1

            if result == sample.label:
                right_case = right_case + 1
            else:
                wrong_case = wrong_case + 1
            classification_matrix[result - 1][sample.label - 1] = classification_matrix[result - 1][
                                                                      sample.label - 1] + 1
        print("N = " + str(n) + " Wrong = " + str(wrong_case) + " Right = " + str(right_case))
        print("Accuarcy = " + str(right_case * 1.0 / n))
        confusion_matrixs = []
        for classifier in self.classifiers:
            confusion_matrixs.append(classifier.get_confusion_matrix())
        macro_P = 0.0
        macro_R = 0.0
        avg_tp = 0.0
        avg_fp = 0.0
        avg_fn = 0.0
        for confusion_matrix in confusion_matrixs:
            macro_P += confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
            macro_R += confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[3])
            avg_tp += confusion_matrix[0]
            avg_fp += confusion_matrix[1]
            avg_fn += confusion_matrix[3]
        macro_P /= 26
        macro_R /= 26
        avg_tp /= 26
        avg_fp /= 26
        avg_fn /= 26
        macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
        micro_P = avg_tp / (avg_tp + avg_fp)
        micro_R = avg_tp / (avg_tp + avg_fn)
        micro_F1 = (2 * micro_P * micro_R) / (micro_P + micro_R)
        print("macro-P: " + str(macro_P))
        print("macro-R: " + str(macro_R))
        print("macro-F1: " + str(macro_F1))
        print("micro-P: " + str(micro_P))
        print("micro-R: " + str(micro_R))
        print("micro-F1: " + str(micro_F1))
        export_classification_matrix(classification_matrix)


# 对样本欠采样，均匀采样生成和正例数目接近的反例
def split(positive_label, num, groups):
    negative_group = Group(-1 * positive_label)
    n = int(num / 25)
    for i in range(0, len(groups)):
        if i != positive_label - 1:
            step = int(groups[i].size() / n)  # 根据步长，均匀采样
            j = 1
            while j < groups[i].size():
                negative_group.add_sample(groups[i].get_sample(j))
                j = j + step
    return negative_group


# 测试单个模型， 获得混淆矩阵
def test_single_classifier(classifier, positive_label):
    test_set = pd.read_csv("test_set.csv", sep=",")
    test_list = test_set.values.tolist()
    test_samples = []
    for i in range(0, len(test_list)):
        features = test_list[i][0:-1]
        label = test_list[i][-1]
        sample = Sample(features, label)
        sample.add_dimension()
        test_samples.append(Sample(features, label))
    n = len(test_samples)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for sample in test_samples:
        logit = classifier.classify(sample)
        if logit > 0:
            if sample.get_label() == positive_label:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if sample.get_label() == positive_label:
                fn = fn + 1
            else:
                tn = tn + 1

    print("n = " + str(n))
    print("TP = " + str(tp) + " Fp = " + str(fp) + " TN = " + str(tn) + " FN = " + str(fn))
    return tp, fp, tn, fn


def get_test_samples():
    test_set = pd.read_csv("test_set.csv")
    test_list = test_set.values.tolist()
    samples = []
    for i in range(0, len(test_list)):
        features = test_list[i][0:-1]
        label = test_list[i][-1]
        sample = Sample(features, label)
        sample.add_dimension()
        samples.append(Sample(features, label))
    return samples


# 从文件加载分类器
def load_classifiers(path):
    fin = open(path, "r")
    _classifiers = []
    for i in range(26):
        line = fin.readline()
        temp_list = line.split(" ")
        beta = []
        for j in range(len(temp_list[0:-1])):
            beta.append(float(temp_list[j]))
        print(beta)
        _classifiers.append(BinaryClassfier(np.array(beta)))
    for i in range(len(_classifiers)):
        confusion_matrix = test_single_classifier(_classifiers[i], i + 1)
        _classifiers[i].set_confusion_matrix(confusion_matrix)
    return _classifiers


def test_exisiting_models(model_path):
    classifiers = load_classifiers(model_path)
    samples = get_test_samples()
    tester = Tester(classifiers, samples)
    tester.test()


def train_and_test():
    train_set = pd.read_csv("train_set.csv", sep=",")
    raw_list = train_set.values.tolist()
    groups = []
    for i in range(1, 27):
        groups.append(Group(i))
    for i in range(0, len(raw_list)):
        features = raw_list[i][0:-1]
        label = raw_list[i][-1]
        sample = Sample(features, label)
        sample.add_dimension()
        groups[label - 1].add_sample(sample)
    samples = get_test_samples()
    trainer = MultiTrainer(groups)
    trainer.generate_classifiers()
    tester = Tester(trainer.get_classifiers(), samples)
    tester.test()


def export_classification_matrix(classification_matrix):
    alpha_list = 'abcdefghijklmnopqrstuvwxyz'
    file = open("classification_matrix.txt", 'w')
    file.write('a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t   u   v   w   x   y   z  <-- classify as\n')
    for i in range(26):
        for j in range(26):
            file.write('%-4s'%str(int(classification_matrix[i][j])))
        file.write(alpha_list[i])
        file.write('\n')
    file.close()


# 使用牛顿法要优于梯度下降，牛顿法收敛极快，准确度高
if __name__ == "__main__":
    #test_exisiting_models("model.txt")
    train_and_test()

# breakcondition = 1e-4 , Accuracy = 72.2%
