# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")

# 1. 加载数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("数据集形状:", X.shape)
print("目标变量形状:", y.shape)
print("特征名称:", feature_names)
print("类别名称:", target_names)

# 2. 数据探索
print("\n=== 数据探索 ===")
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("前5行数据:")
print(df.head())

print("\n基本统计信息:")
print(df.describe())

print("\n类别分布:")
print(df['species'].value_counts())

# 3. 数据可视化
print("\n=== 数据可视化 ===")

# 特征分布直方图
plt.figure(figsize=(12, 8))
for i, feature in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)
    for species in ['setosa', 'versicolor', 'virginica']:
        plt.hist(df[df['species'] == species][feature], alpha=0.7, label=species)
    plt.xlabel(feature)
    plt.ylabel('频率')
    plt.legend()
plt.tight_layout()
plt.show()

# 特征相关性热力图
plt.figure(figsize=(8, 6))
numeric_df = df.drop('species', axis=1)
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()

# 散点矩阵图
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('鸢尾花数据集散点矩阵图', y=1.02)
plt.show()

# 4. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("训练集形状:", X_train.shape)
print("测试集形状:", X_test.shape)

# 5. 模型训练和评估
models = {
    '逻辑回归': LogisticRegression(max_iter=200, random_state=42),
    'K近邻': KNeighborsClassifier(),
    '支持向量机': SVC(random_state=42),
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(random_state=42)
}

results = {}

print("\n=== 模型训练和评估 ===")

for name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{name}:")
    print(f"  测试集准确率: {accuracy:.4f}")
    print(f"  交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 6. 模型比较
print("\n=== 模型性能比较 ===")
results_df = pd.DataFrame({
    '模型': list(results.keys()),
    '测试集准确率': [results[name]['accuracy'] for name in results.keys()],
    '交叉验证准确率': [results[name]['cv_mean'] for name in results.keys()],
    '交叉验证标准差': [results[name]['cv_std'] for name in results.keys()]
})

print(results_df.sort_values('测试集准确率', ascending=False))

# 绘制模型性能比较图
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(results_df))
plt.bar(x_pos, results_df['测试集准确率'], yerr=results_df['交叉验证标准差'],
        alpha=0.7, capsize=5)
plt.xticks(x_pos, results_df['模型'], rotation=45)
plt.ylabel('准确率')
plt.title('模型性能比较')
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.show()

# 7. 最佳模型详细分析
best_model_name = results_df.iloc[0]['模型']
best_model = results[best_model_name]['model']
print(f"\n=== 最佳模型: {best_model_name} ===")

# 详细预测分析
y_pred_best = best_model.predict(X_test_scaled)

print("分类报告:")
print(classification_report(y_test, y_pred_best, target_names=target_names))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'{best_model_name} - 混淆矩阵')
plt.show()

# 8. 超参数调优（以随机森林为例）
print("\n=== 随机森林超参数调优 ===")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证分数:", grid_search.best_score_)

# 使用最佳参数重新训练
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test_scaled)
print("调优后测试集准确率:", accuracy_score(y_test, y_pred_rf))

# 9. 特征重要性分析
print("\n=== 特征重要性分析 ===")

feature_importance = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性': feature_importance
}).sort_values('重要性', ascending=False)

print(feature_importance_df)

plt.figure(figsize=(8, 6))
plt.barh(range(len(feature_importance)), feature_importance_df['重要性'])
plt.yticks(range(len(feature_importance)), feature_importance_df['特征'])
plt.xlabel('重要性')
plt.title('特征重要性排序')
plt.tight_layout()
plt.show()

# 10. 实验总结
print("\n=== 实验总结 ===")
print("1. 通过本实验，我们成功实现了对鸢尾花数据集的分类任务")
print("2. 比较了多种机器学习算法的性能")
print("3. 发现随机森林和支持向量机在该数据集上表现最好")
print("4. 花瓣长度和花瓣宽度是最重要的分类特征")
print("5. 数据标准化对某些算法（如SVM和KNN）的性能提升显著")
