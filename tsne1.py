import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
import os

def generate_data(n_classes=5, n_samples=200, cluster_std=1.0, cluster_sep=2.5, random_state=42):
    """
    生成多簇高维数据。
    - n_classes: 类别数
    - n_samples: 每个类别的样本数
    - cluster_std: 簇的标准差（越小越紧密）
    - cluster_sep: 簇中心间隔倍数（越大簇间距离越大）
    """
    # 随机生成类别中心
    centers = np.random.RandomState(random_state).uniform(
        low=-cluster_sep, high=cluster_sep, size=(n_classes, 2)
    )
    X, y = make_blobs(
        n_samples=n_samples * n_classes,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y

def plot_tsne(X, y, perplexity=30, random_state=0, title="t-SNE Visualization"):
    """
    对高维数据做 t-SNE 并绘制散点图。
    - perplexity: t-SNE 的 perplexity 参数
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_emb = tsne.fit_transform(X)

    # --- 顶会风格美化设置 ---
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Updated rcParams for styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.spines.top": True,      # Show top spine for frame
        "axes.spines.right": True,    # Show right spine for frame
        "axes.spines.left": True,     # Show left spine for frame
        "axes.spines.bottom": True,   # Show bottom spine for frame
        "axes.edgecolor": 'black',    # Set spine color to black
        "axes.linewidth": 1.0,        # Set spine linewidth
        "xtick.major.size": 0,      # No major tick marks
        "ytick.major.size": 0,      # No major tick marks
        "xtick.minor.size": 0,      # No minor tick marks
        "ytick.minor.size": 0,      # No minor tick marks
        "xtick.labelbottom": False, # No x-axis tick labels (redundant with plt.xticks([]))
        "ytick.labelleft": False,   # No y-axis tick labels (redundant with plt.yticks([]))
    })

    # 1. 隐藏xy轴的刻度和标签 (Hide xy axis ticks and labels)
    plt.xticks([])
    plt.yticks([])

    # 调色板：Updated color palette based on the provided image reference
    # Selected 5 distinct colors from the bar chart image:
    # 1. Orange/Beige: #D89F7B
    # 2. Pink/Rose: #E0A8B2
    # 3. Muted Purple: #A59EC4
    # 4. Desaturated Blue: #9DB6D0
    # 5. Grayish Teal/Blue-green: #739F97
    colors = ['#D89F7B', '#E0A8B2', '#A59EC4', '#9DB6D0', '#739F97']


    for cls in np.unique(y):
        idx = (y == cls)
        plt.scatter(
            X_emb[idx, 0], X_emb[idx, 1],
            s=20,                    # 点大小
            c=colors[int(cls) % len(colors)],
            label=f"Class {int(cls)+1}",
            alpha=0.8,               # 半透明效果
            edgecolors='none'
        )

    # plt.title(title, pad=15) # Title is currently commented out
    plt.legend(frameon=True, loc='best') # Keep legend with a frame
    plt.tight_layout()
    
    # Ensure the 'plots' directory exists
    plot_dir = "plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")

    plt.savefig(os.path.join(plot_dir, "tsne_visualization.pdf"), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(plot_dir, 'tsne_visualization.pdf')}")
    plt.show()


if __name__ == "__main__":
    # 超参数设置
    n_samples_per_class = 1000   # 每类样本数
    cluster_std = 1.2            # 簇内紧密程度（标准差）
    cluster_sep = 5            # 簇中心间隔
    perplexity = 40              # t-SNE perplexity

    # 生成数据并可视化
    X, y = generate_data(
        n_classes=5,
        n_samples=n_samples_per_class,
        cluster_std=cluster_std,
        cluster_sep=cluster_sep,
        random_state=3 # Original random_state for data generation
    )
    plot_tsne(
        X, y,
        perplexity=perplexity,
        random_state=123, # Original random_state for t-SNE
        title="五分类 t-SNE 可视化示意图" # Title is passed but currently commented out in plot_tsne
    )