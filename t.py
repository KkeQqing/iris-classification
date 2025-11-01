import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------- 1. 加载并探索鸢尾花数据集 ----------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 转换为DataFrame以便探索
iris_df = pd.DataFrame(data=np.c_[X, y], columns=feature_names + ['species'])
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("数据集基本信息：")
iris_df.info()
print("\n数据集前5行：")
print(iris_df.head())
print("\n各物种分布：")
print(iris_df['species'].value_counts())

# 可视化探索：特征两两分布
sns.pairplot(iris_df, hue='species')
plt.suptitle('Iris Feature Pairplot', y=1.02)
plt.show()

# ---------------------- 2. 数据预处理与划分 ----------------------
# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------- 3. 多种分类算法实现与参数调优 ----------------------
# 定义算法列表及参数网格
models = {
    'Logistic Regression': (
        LogisticRegression(),
        {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    ),
    'Random Forest': (
        RandomForestClassifier(),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
    ),
    'SVM': (
        SVC(),
        {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    )
}

# 训练并评估各模型
results = {}
for name, (model, param_grid) in models.items():
    # 网格搜索参数调优
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # 在测试集上评估
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'best_model': best_model,
        'best_params': best_params,
        'cv_accuracy': best_score,
        'test_accuracy': test_accuracy,
        'classification_report': classification_report(y_test, y_pred, target_names=target_names)
    }

    print(f"\n=== {name} 结果 ===")
    print(f"最佳参数: {best_params}")
    print(f"交叉验证准确率: {best_score:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"分类报告:\n{results[name]['classification_report']}")

# ---------------------- 4. 模型性能比较与可视化 ----------------------
# 整理各模型准确率
model_names = list(results.keys())
cv_acc = [results[name]['cv_accuracy'] for name in model_names]
test_acc = [results[name]['test_accuracy'] for name in model_names]

# 绘制准确率对比条形图
x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, cv_acc, width, label='交叉验证准确率')
rects2 = ax.bar(x + width / 2, test_acc, width, label='测试集准确率')

ax.set_ylabel('准确率')
ax.set_title('不同分类算法性能对比')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()


# 标注数值
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()

# 绘制混淆矩阵（以最佳模型为例，这里选随机森林）
best_model_name = max(results, key=lambda k: results[k]['test_accuracy'])
best_model = results[best_model_name]['best_model']
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title(f'{best_model_name} 混淆矩阵')
plt.show()

# ---------------------- 5. 特征重要性分析（以随机森林为例） ----------------------
if 'Random Forest' in results:
    rf_model = results['Random Fores' ]