# Adjusting the standalone color bar's height to match the main plot's height from the previous visualization
# and unifying the font size.

# Assuming the main plot's height was approximately 6 inches based on standard figsize ratios,
# we adjust the height of the color bar accordingly.

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Define the font size to match the main plot
font_size = 14  # Adjust as needed to match the main plot
plt.rcParams.update({'font.size': font_size})

# Define the colormap
cmap = mpl.cm.coolwarm

# Create a figure for the color bar that matches the main plot's height
fig, ax = plt.subplots(figsize=(1, 10))  # Width of 2 inches, height of 6 inches to match the main plot
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=mpl.colors.Normalize(vmin=-1, vmax=1),
                                ticks=np.arange(-1, 1.1, 0.2),
                                orientation='vertical')

# Set the color bar label
cb1.set_label('Score', fontsize=font_size)
cb1.ax.tick_params(labelsize=font_size)  # Ensure tick labels match the font size

# 保存为SVG文件
svg_file_path = './train_log/GrasshopperEnv_True/color_bar.svg'  # 指定保存位置
plt.savefig(svg_file_path, format='svg')

plt.show()

svg_file_path  # 返回文件路径以供下载
