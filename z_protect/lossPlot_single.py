import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the loss data
file_path = 'E:/project/AI_Designer/PARL_01/examples/MADDPG_16_hyper_vision/z_protect/加强绕路和不动扣分/train_log/Park_Env_True/run-train_log_Park_Env_True-tag-Loss_Critic_Loss.csv'

loss_data = pd.read_csv(file_path)


mpl.rcParams['font.size'] = 24  # 全局字体大小


# Apply moving average to smooth the loss curves
def moving_average(data, window_size=50):
    return data.rolling(window=window_size, min_periods=1).mean()

# Extract Step and Value for Hybrid network
steps = loss_data['Step']
hybrid_smoothed = moving_average(loss_data['Value'])


# Plot the smoothed loss curves with distinct colors and added transparency
plt.figure(figsize=(10, 6))
plt.plot(steps, hybrid_smoothed, label='Hybrid Network Loss (Smoothed)', color='red', linewidth=2, alpha=0.8)
plt.xlabel('Training Episode', fontsize=24)
plt.ylabel('Loss Value', fontsize=24)
plt.title('Smoothed Loss Curve', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=24)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)

# Save the smoothed comparison plot as an SVG file
transparent_output_path = './smoothed_loss_curve_comparison_transparentSingle.svg'
plt.savefig(transparent_output_path, format='svg')
plt.show()

print(f"Plot saved as SVG format at: {transparent_output_path}")
