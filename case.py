import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import argparse

def plot_expert_distribution_comparison(llm_selections_path, moe_selections_path, num_experts, output_path="expert_distribution_comparison.png"):
    """
    绘制 LLM 和 MoE Router 专家选择的分布对比图。
    """
    try:
        # 加载 LLM 选择 (假设针对源域所有节点)
        with open(llm_selections_path, 'r') as f:
            llm_selections_dict = json.load(f)
        # 注意：LLM 选择是针对源域的，MoE 是针对目标域的，直接对比可能意义有限
        # 这里我们假设对比的是 LLM 对源域节点的先验选择 vs MoE 对目标域节点的最终选择
        llm_choices = list(llm_selections_dict.values())
        llm_counts = np.bincount(llm_choices, minlength=num_experts)
        llm_dist = llm_counts / llm_counts.sum() if llm_counts.sum() > 0 else np.zeros(num_experts)

        # 加载 MoE 选择 (针对目标域节点)
        moe_df = pd.read_csv(moe_selections_path)
        # 假设我们关心每个节点概率最高的那个专家选择
        # 注意：原始 CSV 保存的是 top-k，这里简化为取第一个（通常 k=1）
        moe_choices = moe_df.groupby('Node ID')['Selected Expert'].first().tolist() # 或者基于 'Probability' 选择
        moe_counts = np.bincount(moe_choices, minlength=num_experts)
        moe_dist = moe_counts / moe_counts.sum() if moe_counts.sum() > 0 else np.zeros(num_experts)

        # 绘图
        labels = [f'Expert {i}' for i in range(num_experts)]
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, llm_dist, width, label=f'LLM Prior ({args.source} nodes)', color='skyblue')
        rects2 = ax.bar(x + width/2, moe_dist, width, label=f'MoE Router ({args.target} nodes)', color='lightcoral')

        ax.set_ylabel('Proportion')
        ax.set_title('Expert Selection Distribution: LLM Prior vs MoE Router')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')

        fig.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Expert distribution comparison plot saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find input files: {llm_selections_path} or {moe_selections_path}")
    except Exception as e:
        print(f"An error occurred during distribution plotting: {e}")

def plot_tsne_visualization(embeddings_path, labels_path, output_path="tsne_visualization.png"):
    """
    绘制 MoE 模型最终输出嵌入的 t-SNE 可视化图。
    """
    try:
        embeddings = torch.load(embeddings_path).detach().numpy()
        labels = torch.load(labels_path).detach().numpy()

        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=250) # 调整参数
        embeddings_2d = tsne.fit_transform(embeddings)
        print("t-SNE finished.")

        # 绘图
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            hue=labels,
            palette=sns.color_palette("hsv", len(np.unique(labels))),
            legend="full",
            alpha=0.7
        )
        plt.title(f't-SNE Visualization of Target Node Embeddings ({args.target})')
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.savefig(output_path)
        plt.close()
        print(f"t-SNE visualization saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Could not find input files: {embeddings_path} or {labels_path}")
    except Exception as e:
        print(f"An error occurred during t-SNE plotting: {e}")

if __name__ == "__main__":
    # --- 从 main.py 获取参数 (或者直接硬编码) ---
    # 您需要提供与运行 main.py 时相同的 source, target, llm, expert_num 等参数
    parser_analysis = argparse.ArgumentParser()
    parser_analysis.add_argument("--source", type=str, default='dblpv7', help="Source domain name from main.py")
    parser_analysis.add_argument("--target", type=str, default='citationv1', help="Target domain name from main.py")
    parser_analysis.add_argument("--llm", type=str, default='qwen2.5:7b', help="LLM model name used in main.py")
    parser_analysis.add_argument("--expert_num", type=int, default=3, help="Number of experts used in main.py")
    args = parser_analysis.parse_args()

    # 构建文件路径
    llm_selections_file = f"log/{args.source}-{args.target}-{args.llm}-selections.json"
    moe_selections_file = f"log/{args.target}-expert-selection.csv"
    embeddings_file = f"log/{args.target}-final-embeddings.pt" # 与 main.py 中修改的保存路径一致
    labels_file = f"log/{args.target}-final-labels.pt" # 与 main.py 中修改的保存路径一致

    # --- 执行绘图 ---
    plot_expert_distribution_comparison(llm_selections_file, moe_selections_file, args.expert_num)
    plot_tsne_visualization(embeddings_file, labels_file)

    print("Case study analysis finished.")