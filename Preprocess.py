# 第三方库
import pandas as pd


# 一个样本
class Sample:
    # 构造函数， 初始化特征和标签
    def __init__(self, features, label):
        self.features = features
        self.label = label

    def display(self):
        print("features: " + str(self.features) + " label: " + str(self.label))

    def get_features(self):
        return self.features

    # 方便处理，构造 [x, 1]
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


# 一组训练数据集： 包含正例和反例
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


# 从文件获取测试样本
def get_test_samples(path):
    test_set = pd.read_csv(path)
    test_list = test_set.values.tolist()
    samples = []
    for i in range(0, len(test_list)):
        features = test_list[i][0:-1]
        label = test_list[i][-1]
        sample = Sample(features, label)
        sample.add_dimension()
        samples.append(Sample(features, label))
    return samples


# 从文件获取训练样本，并按照标签分组
def get_train_sample_groups(path):
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
    return groups


# 对反例样本欠采样，均匀采样生成指定数目的训练集
def split(positive_label, num, groups):
    negative_group = Group(-1 * positive_label)
    # 每一个字母取1/25
    n = int(num / 25)
    for i in range(0, len(groups)):
        if i != positive_label - 1:
            step = int(groups[i].size() / n)  # 根据步长，均匀采样
            if step == 0:
                step = 1
            j = 1
            while j < groups[i].size():
                negative_group.add_sample(groups[i].get_sample(j))
                j = j + step
    return negative_group


