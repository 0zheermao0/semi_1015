import matplotlib.pyplot as plt
import numpy as np

# 估计的 K 和 p 标签
p_labels = [0.01, 0.03, 0.05, 0.07, 0.09] # x axis
k_labels = [0.2, 0.4, 0.6, 0.8, 1] # y axis

# 对应于 k_labels 和 p_labels 的数值位置
k_pos = np.arange(len(k_labels))
p_pos = np.arange(len(p_labels))

# 估计的 Z 值 (高度) - 请用您的真实数据替换
# 顺序是：对每个 k，遍历所有的 p
data_z = np.array([
    # K = 0.01 (corresponds to first k_label) changing y
    0.6611, 0.6700, 0.6899, 0.6933, 0.6844,
    # K = 0.03
    0.6722, 0.6833, 0.6944, 0.708, 0.6955,
    # K = 0.05
    0.6777, 0.6866, 0.6955, 0.707, 0.6966,
    # K = 0.07
    0.6800, 0.6999, 0.6888, 0.6911, 0.6877,
    # K = 0.09
    0.6533, 0.6622, 0.6711, 0.6822, 0.6788,
])

# 创建 X, Y 坐标网格
# bar3d 需要的是每个条形的左下角坐标
xpos, ypos = np.meshgrid(k_pos, p_pos, indexing="ij") # 'ij' 使得 k 对应行，p 对应列
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos) # 条形的底部都在 z=0

# 条形的宽度和深度
dx = 0.8  # x 方向的宽度
dy = 0.8  # y 方向的深度
dz = data_z # 条形的高度

# 颜色 (每个 K 值一组颜色)
# Corrected comments to match k_labels
colors_k_groups = [
    '#ADD8E6',  # 浅蓝色 (Light Blue) for K=0.01
    '#90EE90',  # 浅绿色 (Light Green) for K=0.03
    '#FFFFE0',  # 亮黄色 (Light Yellow) for K=0.05
    '#FFB6C1',  # 浅粉色 (Light Pink) for K=0.07
    '#E6E6FA',  # 淡紫色 (Lavender) for K=0.09
]

# 为每个条形分配颜色
bar_colors = []
for i in range(len(k_labels)):
    bar_colors.extend([colors_k_groups[i]] * len(p_labels))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colors, edgecolor='black', linewidth=0.5, shade=True) # Added shade for better 3D effect

# 设置坐标轴标签
ax.set_xlabel('smoothing coefficient $\\alpha$', fontsize=16, labelpad=10) # Added labelpad
ax.set_ylabel('activation threshold $\\theta$', fontsize=16, labelpad=10) # Added labelpad
ax.set_zlabel('Micro F1', fontsize=16, labelpad=15) # Increased labelpad for Z axis

# 设置刻度标签
ax.set_xticks(k_pos + dx / 2) # 将刻度放在条形中间
ax.set_xticklabels(k_labels, fontsize=14)

ax.set_yticks(p_pos + dy / 2) # 将刻度放在条形中间
ax.set_yticklabels(p_labels, fontsize=14)

# 设置 Z 轴的范围和刻度
ax.set_zlim(0, 1.0) # Adjusted zlim slightly based on data, max data is ~0.8
ax.set_zticks(np.arange(0, 0.81, 0.2)) # Adjusted zticks to go up to 1.0
ax.tick_params(axis='z', labelsize=14, pad=8) # Adjust Z tick label padding and size

# ax.set_zticklabels([f"{tick:.1f}" for tick in np.arange(0, 1.1, 0.2)], fontsize=14) # Original, replaced by tick_params for finer control if needed

# 设置标题
ax.set_title('Micro F1', fontsize=16, loc='center', pad=20) # Adjusted pad if title were used

# 调整视角
ax.view_init(elev=25, azim=-45) # Original: elev=25, azim=-135+90 = -45

# 添加网格
ax.grid(True)

# Adjust layout to prevent labels from being cut off and reduce whitespace
plt.tight_layout(pad=2.0) # Add padding to tight_layout

# Create plots directory if it doesn't exist
import os
if not os.path.exists("plots"):
    os.makedirs("plots")

plt.savefig("plots/3d_bar_plot.pdf", dpi=300, bbox_inches='tight') # Saving with bbox_inches='tight' is good
plt.show()