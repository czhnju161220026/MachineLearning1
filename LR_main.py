from Preprocess import get_test_samples
from Preprocess import get_train_sample_groups
from Tester import Tester
from Trainer import BinaryClassfier
from Trainer import MultiTrainer
# 第三方库
import numpy as np

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
    return _classifiers


# 测试已有的模型， 参数为模型路径和测试集文件路径
def test_exisiting_models(model_path, test_set_path):
    # 加载分类器
    classifiers = load_classifiers(model_path)
    # 获得测试集样本
    samples = get_test_samples(test_set_path)
    # 创建Tester对象，并开始测试
    tester = Tester(classifiers, samples)
    tester.test()


# 训练并测试，参数为训练集文件路径和测试集文件路径
def train_and_test(train_set_path, test_set_path):
    # 获得训练集样本，并按标签分类
    groups = get_train_sample_groups(train_set_path)
    # 获得测试即样本
    samples = get_test_samples(test_set_path)
    # 创建一个MultiTrainer对象，训练多个分类器
    trainer = MultiTrainer(groups)
    trainer.generate_classifiers()
    # 创建一个Tester对象，并测试分类器
    tester = Tester(trainer.get_classifiers(), samples)
    tester.test()


if __name__ == "__main__":
    # 从文件中加载模型，并测试
    # test_exisiting_models("models.txt", "test_set.csv")
    # 训练模型，并测试。测试后生成模型文件
    train_and_test("train_set.csv", "test_set.csv")

# breakcondition = 1e-4 , Accuracy = 72.2%
