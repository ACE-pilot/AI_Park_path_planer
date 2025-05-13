import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

# Define the size of the area (assuming square for simplicity)
size = 8  # in meters
x = np.linspace(0, size, 100)
y = np.linspace(0, size, 100)
X, Y = np.meshgrid(x, y)

# Define the positions and width of the columns
columns = [(0, 0), (0, size), (size, 0), (size, size)]
column_width = 0.8  # meters

# Function to calculate the cubic Bezier curve point at t using given control points
def cubic_bezier(t, points):
    n = 3  # Degree of the curve
    return sum(comb(n, i) * (1-t)**(n-i) * t**i * points[i] for i in range(n+1))

# Bezier curve control points for scoring
control_points = np.array([[0, 1], [size/3, -1], [2*size/3, -1], [size, 1]])
t_values = np.linspace(0, 1, 100)
bezier_points = np.array([cubic_bezier(t, control_points) for t in t_values])
bezier_x = bezier_points[:, 0]
bezier_y = bezier_points[:, 1]

# Function to calculate distance to the nearest column
def distance_to_nearest_column(x, y, columns):
    return min([np.sqrt((x - col[0])**2 + (y - col[1])**2) for col in columns])

# Function to interpolate the Bezier curve for the scoring system
def interpolate_bezier(distance, max_distance, bezier_x, bezier_y):
    t = distance / max_distance
    index = np.argmin(np.abs(bezier_x - t * size))
    return bezier_y[index]

# Calculate distance for each point in the grid
distances = np.array([[distance_to_nearest_column(xi, yi, columns) for xi in x] for yi in y])

# Normalize distances
max_distance = np.sqrt(2) * size
normalized_distances = distances / max_distance

# Apply the Bezier curve for color mapping
bezier_scores = np.array([[interpolate_bezier(dist, max_distance, bezier_x, bezier_y) for dist in row] for row in distances])
bezier_y_normalized = (bezier_y - min(bezier_y)) / (max(bezier_y) - min(bezier_y)) * 2 - 1

# Set the font size
plt.rcParams.update({'font.size': 16})

# Create the main plot with additional axes for the scoring functions and a narrower color bar
fig = plt.figure(figsize=(16, 14))
ax_main = plt.subplot2grid((4, 5), (1, 0), colspan=3, rowspan=3)
ax_top = plt.subplot2grid((4, 5), (0, 0), colspan=3, sharex=ax_main)
ax_right = plt.subplot2grid((4, 5), (1, 3), rowspan=3, sharey=ax_main)
cbar_ax = plt.subplot2grid((4, 5), (1, 4), rowspan=3)

# Main plot with the Bezier color map
color_mapping = ax_main.pcolor(X, Y, bezier_scores, cmap=plt.cm.coolwarm, shading='auto')
cbar = fig.colorbar(color_mapping, cax=cbar_ax, orientation='vertical')
cbar_ax.set_ylabel('Bezier Score', fontsize=18)
cbar.ax.tick_params(labelsize=14)

# Plot the columns as squares
for col in columns:
    square = plt.Rectangle((col[0] - column_width/2, col[1] - column_width/2), 
                           column_width, column_width, color='black')
    ax_main.add_patch(square)

# Plotting and filling the Bezier scoring function on the right (for Y-axis)
ax_right.plot(bezier_y_normalized, bezier_x, 'k-')
ax_right.fill_betweenx(bezier_x, bezier_y_normalized, 0, where=(bezier_y_normalized > 0), color='red', alpha=0.3)
ax_right.fill_betweenx(bezier_x, bezier_y_normalized, 0, where=(bezier_y_normalized < 0), color='blue', alpha=0.3)
ax_right.set_xlim(-1, 1)
ax_right.set_ylim(0, size)
ax_right.set_xlabel('Score', fontsize=20)
ax_right.set_ylabel('Distance (Y)', fontsize=20)
ax_right.yaxis.set_label_position('right')
ax_right.axvline(0, color='gray', linestyle='--')

# Plotting and filling the Bezier scoring function on the top (for X-axis)
ax_top.plot(bezier_x, bezier_y_normalized, 'k-')
ax_top.fill_between(bezier_x, bezier_y_normalized, 0, where=(bezier_y_normalized > 0), color='red', alpha=0.3)
ax_top.fill_between(bezier_x, bezier_y_normalized, 0, where=(bezier_y_normalized < 0), color='blue', alpha=0.3)
ax_top.set_xlim(0, size)
ax_top.set_ylim(-1, 1)
ax_top.xaxis.tick_top()
ax_top.xaxis.set_label_position('top')
ax_top.set_ylabel('Score', fontsize=20)
ax_top.set_xlabel('Distance (X)', fontsize=20)
ax_top.axhline(0, color='gray', linestyle='--')

# Setting titles and labels for the main plot
ax_main.set_title("Color Map with Normalized Bezier Scoring Functions", fontsize=22)
ax_main.set_xlabel("Distance in meters (X)", fontsize=20)
ax_main.set_ylabel("Distance in meters (Y)", fontsize=20)
ax_main.axis([0, size, 0, size])
ax_main.set_aspect('equal', adjustable='box')

plt.tight_layout()

# 保存为SVG文件
svg_file_path = './train_log/GrasshopperEnv_True/column_analysis.svg'  # 指定保存位置
plt.savefig(svg_file_path, format='svg')

plt.show()

svg_file_path  # 返回文件路径以供下载

