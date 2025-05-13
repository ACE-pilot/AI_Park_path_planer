import pandas as pd
import matplotlib.pyplot as plt

# Load the loss data
file_path = 'E:/project/AI_Designer/PARL_01/examples/MADDPG_16_hyper_vision/z_protect/加强绕路和不动扣分/train_log/Park_Env_True/run-train_log_Park_Env_True-tag-train_episode_reward_wrt_episode.csv'
mlp_file_path = 'E:/project/AI_Designer/PARL_01/examples/MADDPG_16_MLP_vision2/z_protect/train_log/Park_Env_True/run-train_log_Park_Env_True-tag-train_episode_reward_wrt_episode.csv'

loss_data = pd.read_csv(file_path)
mlp_loss_data = pd.read_csv(mlp_file_path)

# Apply moving average to smooth the loss curves
def moving_average(data, window_size=50):
    return data.rolling(window=window_size, min_periods=1).mean()

# Extract Step and Value for Hybrid network
steps = loss_data['Step']
hybrid_smoothed = moving_average(loss_data['Value'])

# Extract Step and Value for MLP network
mlp_steps = mlp_loss_data['Step']
mlp_smoothed = moving_average(mlp_loss_data['Value'])

# Plot the smoothed loss curves with distinct colors and added transparency
plt.figure(figsize=(10, 6))
plt.plot(steps, hybrid_smoothed, label='Hybrid Network Loss (Smoothed)', color='red', linewidth=2, alpha=0.8)
plt.plot(mlp_steps, mlp_smoothed, label='MLP Network Loss (Smoothed)', color='blue', linewidth=2, alpha=0.8)
plt.xlabel('Training Episode', fontsize=12)
plt.ylabel('Reward Value', fontsize=12)
plt.title('Smoothed Episode Reward Curve Comparison: Hybrid vs MLP Networks', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)

# Save the smoothed comparison plot as an SVG file
transparent_output_path = './smoothed_episode_reward_curve_comparison_transparent.svg'
plt.savefig(transparent_output_path, format='svg')
plt.show()

print(f"Plot saved as SVG format at: {transparent_output_path}")
