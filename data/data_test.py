import scipy.io as sio
import scipy.sparse  # 添加这一行
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# 加载.mat文件（替换为实际路径）
file_path = './data/acmv9.mat'  # 修改为实际路径
net = sio.loadmat(file_path)
# check all keys
for key in net.keys():
    print(key)

# 提取数据
x = net['attrb']
edge_index = net['network']
y = net['group']

# 打印原始标签形状
print("原始标签形状：", y.shape)
# 确保标签是一维数组
y = y.reshape(-1)
if y.shape[0] != x.shape[0]:
    print("警告：标签数量与节点数量不匹配！")
    # 如果标签是one-hot形式，转换为类别索引
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    # 如果仍然不匹配，只使用前 x.shape[0] 个标签
    if y.shape[0] > x.shape[0]:
        y = y[:x.shape[0]]

# 打印基本信息
print("="*40)
print("数据基本信息：")
print(f"节点特征矩阵 x 形状：{x.shape} (节点数 × 特征维度)")
print(f"边信息 edge_index 形状：{edge_index.shape}")
print(f"标签 y 形状：{y.shape}")

# 检查是否为稀疏矩阵（MATLAB稀疏矩阵会被scipy转换为csr_matrix）
if hasattr(edge_index, 'indices'):
    print("\n检测到稀疏矩阵，转换为稠密矩阵...")
    edge_index = edge_index.toarray()

# 处理邻接矩阵（如果是N×N矩阵）
if edge_index.shape[0] == edge_index.shape[1]:
    print("\n检测到邻接矩阵，转换为边列表格式...")
    # 修复：对稀疏矩阵使用正确的方法获取非零元素的索引
    if scipy.sparse.issparse(edge_index):
        rows, cols = edge_index.nonzero()
    else:
        rows, cols = np.nonzero(edge_index)
    edge_index = np.vstack([rows, cols])

print("\n边的数量：", edge_index.shape[1])
print("平均节点度数：", 2*edge_index.shape[1]/x.shape[0])

# 展示部分数据
print("\n" + "="*40)
print("节点特征示例（前5行）：")
print(x[:5])

print("\n边示例（前5条）：")
print(edge_index[:, :5].T)  # 转置为(E,2)格式显示

print("\n标签示例（前10个）：")
print(y[:10].flatten())  # 假设标签是单列的

# 可视化部分
print("\n" + "="*40)
print("数据可视化：")

# 1. 绘制标签分布
plt.figure(figsize=(10, 4))
plt.hist(y, bins=np.unique(y).size)
plt.title("Label Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("label_distribution.png")

# 2. 特征分布（第一维特征）
plt.figure(figsize=(10, 4))
plt.hist(x[:, 0], bins=50)
plt.title("First Feature Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Count")
plt.savefig("feature_distribution.png")

# 3. PCA可视化（如果特征维度>2）
if x.shape[1] > 2:
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(y)
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap='viridis', 
                         s=20, alpha=0.6, vmin=min(unique_labels), vmax=max(unique_labels))
    plt.colorbar(scatter)
    plt.title("PCA Visualization of Node Features")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("pca_visualization.png")

# 4. 绘制子图（抽样部分节点避免内存问题）
sample_nodes = np.random.choice(x.shape[0], size=200, replace=False)
node_mask = np.isin(edge_index[0], sample_nodes) & np.isin(edge_index[1], sample_nodes)
sampled_edges = edge_index[:, node_mask]

G = nx.Graph()
G.add_edges_from(sampled_edges.T)
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(10, 8))
nx.draw(G, pos, 
        node_size=30, 
        width=0.5, 
        node_color='skyblue', 
        with_labels=False, 
        edge_color='gray')
plt.title("Sampled Graph Visualization")
plt.savefig("sampled_graph.png")
