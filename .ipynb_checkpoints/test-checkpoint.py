# 数据处理
import numpy as np
import pandas as pd

# 机器学习
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === 解决中文显示问题 ===
import matplotlib.pyplot as plt

# 设置中文字体（Windows 系统常用 'SimHei'，即黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号（如坐标轴上的 -1）
# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
# 加载鸢尾花数据集
iris = datasets.load_iris()

# 查看基本信息
print("特征名称:", iris.feature_names)
print("类别名称:", iris.target_names)
print("数据形状:", iris.data.shape)  # (150, 4)
print("标签形状:", iris.target.shape)  # (150,)

X = iris.data      # 特征矩阵 (150, 4)
y = iris.target    # 标签向量 (150,)

# 按 8:2 划分，stratify=y 保证各类别比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持每类在训练/测试中比例相同
)

# 创建 SVM 分类器（使用 RBF 核）
svm_model = SVC(
    kernel='rbf',     # 核函数：径向基函数（适合非线性）
    C=1.0,            # 正则化参数：控制误分类惩罚（默认1.0）
    gamma='scale',    # RBF 核参数：gamma = 1 / (n_features * X.var())
    random_state=42   # 保证结果可复现
)

# 训练模型
svm_model.fit(X_train, y_train)
# 对测试集进行预测
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.2%}")  # 通常为 96.67% 或 100%
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('混淆矩阵')
plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.show()

# 仅使用前两个特征
X_2d = iris.data[:, :2]
y_2d = iris.target

# 重新划分（保持一致性）
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_2d, test_size=0.2, random_state=42, stratify=y_2d
)

# 训练 2D SVM
svm_2d = SVC(kernel='rbf', C=1.0)
svm_2d.fit(X_train_2d, y_train_2d)

# 绘制决策边界
def plot_decision_boundary_2d(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=60)
    plt.colorbar(scatter)
    plt.xlabel('花萼长度 (cm)')
    plt.ylabel('花萼宽度 (cm)')
    plt.title(title)
    plt.show()

plot_decision_boundary_2d(svm_2d, X_2d, y_2d, 'SVM 决策边界（2D 投影）')