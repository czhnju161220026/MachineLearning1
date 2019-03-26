from Preprocess import Sample
from Trainer import BinaryClassfier
# 第三方库
import numpy as np


# 测试单个模型， 获得混淆矩阵
def test_single_classifier(classifier, positive_label, test_samples):
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
    return tp, fp, tn, fn


# 打印最终的混淆矩阵
def export_classification_matrix(classification_matrix):
    alpha_list = 'abcdefghijklmnopqrstuvwxyz'
    file = open("classification_matrix.txt", 'w')
    file.write(
        'a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t   u   v   w   x   y   z  <-- classify as\n')
    for i in range(26):
        for j in range(26):
            file.write('%-4s' % str(int(classification_matrix[i][j])))
        file.write(alpha_list[i])
        file.write('\n')
    file.close()
    print(
        'a   b   c   d   e   f   g   h   i   j   k   l   m   n   o   p   q   r   s   t   u   v   w   x   y   z  <-- classify as')
    for i in range(26):
        for j in range(26):
            print('%-4s' % str(int(classification_matrix[i][j])), end='')
        print(alpha_list[i])


# 测试者类
class Tester:
    # 读入所有的分类器和测试样本
    def __init__(self, _classifiers, _test_samples):
        self.classifiers = _classifiers
        self.test_samples = _test_samples
        # 测试单个分类器的表现
        for i in range(len(self.classifiers)):
            self.classifiers[i].set_confusion_matrix(
                test_single_classifier(self.classifiers[i], i + 1, self.test_samples))

    # 测试多分类的表现
    def test(self):
        n = len(self.test_samples)
        right_case = 0
        wrong_case = 0
        classification_matrix = np.zeros((26, 26))
        for sample in self.test_samples:
            results = []
            for classifier in self.classifiers:
                # 统计每个分类器输出的对数概率 与 准确率 的乘积，作为最终的判别标准
                results.append(classifier.classify(sample) * classifier.get_accuracy())
            result = np.argmax(np.array(results)) + 1

            if result == sample.label:
                right_case = right_case + 1
            else:
                wrong_case = wrong_case + 1
            classification_matrix[result - 1][sample.label - 1] = classification_matrix[result - 1][
                                                                      sample.label - 1] + 1
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
        print("-----------------------------------* Result *------------------------------------")
        export_classification_matrix(classification_matrix)
        print("N = " + str(n) + " Wrong = " + str(wrong_case) + " Right = " + str(right_case))
        print("Accuarcy = %.4f" % (right_case * 1.0 / n))
        print("macro-P: %.4f" % macro_P)
        print("macro-R: %.4f" % macro_R)
        print("macro-F1: %.4f" % macro_F1)
        print("micro-P: %.4f" % micro_P)
        print("micro-R: %.4f" % micro_R)
        print("micro-F1: %.4f" % micro_F1)
