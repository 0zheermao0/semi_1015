import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

# -----------------------------
# 可调节超参数
# -----------------------------
n_samples_per_class = 500        # 每类样本数量
cluster_std = 1.5                # 簇内标准差（越小越紧密）
distance_between_clusters = 4.5   # 簇中心间距（越大簇越分离）
n_classes = 5                    # 分类数
n_features = 10                  # 高维空间维度
random_state = 42                # 随机种子
perplexity = 50                  # t-SNE 的 perplexity 参数

# -----------------------------
# 1. 生成高维合成数据
# -----------------------------
centers = np.zeros((n_classes, n_features))
for i in range(n_classes):
    centers[i, i] = distance_between_clusters  # 每个簇位于不同的轴上

X, y = make_blobs(
    n_samples=n_classes * n_samples_per_class,
    centers=centers,
    cluster_std=cluster_std,
    random_state=random_state,
    shuffle=True
)

# -----------------------------
# 2. t-SNE 降维
# -----------------------------
tsne = TSNE(
    n_components=2,
    random_state=random_state,
    perplexity=perplexity,
    n_iter=500,
    learning_rate='auto'
)
X_tsne = tsne.fit_transform(X)

# -----------------------------
# 3. 数据准备用于绘图
# -----------------------------
df = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
df['Class'] = y

# -----------------------------
# 4. 可视化设置（顶会风格）
# -----------------------------
sns.set(style='white', context='talk')
plt.figure(figsize=(10, 8))

# 使用 colorblind 安全配色方案
palette = sns.color_palette("colorblind", n_classes)

# 绘图
scatter = sns.scatterplot(
    data=df,
    x='Dim1',
    y='Dim2',
    hue='Class',
    palette=palette,
    s=80,
    alpha=0.8,
    edgecolor='none',
    legend='full'
)

# 图例位置调整
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# 坐标轴设置
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)
plt.title('', fontsize=16)

# 去除边框
sns.despine(left=True, bottom=True)

# 自动调整布局
plt.tight_layout()

# 显示图像
plt.show()
plt.savefig('plots/tsne_plot.pdf', dpi=300, bbox_inches='tight')