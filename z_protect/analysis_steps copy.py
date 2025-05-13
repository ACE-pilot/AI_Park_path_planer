import pandas as pd
import matplotlib.pyplot as plt

def read_and_plot_csv(csv_file_path, episode_number):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 过滤出指定 episode 的数据
    episode_data = df[df['episode'] == episode_number]

    # 创建一个 3x3 的图表矩阵
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Episode {episode_number} Analysis", fontsize=16)

    # 遍历每个智能体（3个智能体）
    for i in range(3):
        # 提取智能体 i 的动作和奖励
        actions = episode_data[f"agent_{i}_actions"].values[0].split(";")
        rewards = episode_data[f"agent_{i}_rewards"].values[0].split(";")
        
        # 将每个动作字符串分割为 X 和 Y 的速度变化
        x_speed = []
        y_speed = []
        for action in actions:
            action_values = action.split(",")
            x_speed.append(float(action_values[1]) - float(action_values[0]))  # 右 - 左
            y_speed.append(float(action_values[3]) - float(action_values[2]))  # 下 - 上

        # 将奖励转化为浮动值
        rewards = [float(r) for r in rewards]

        # 绘制子图
        # 第一列: X轴速度变化
        axs[i, 0].plot(x_speed)
        axs[i, 0].set_title(f"Agent {i+1} - X Speed Change")
        axs[i, 0].set_xlabel("Step")
        axs[i, 0].set_ylabel("X Speed")

        # 第二列: Y轴速度变化
        axs[i, 1].plot(y_speed)
        axs[i, 1].set_title(f"Agent {i+1} - Y Speed Change")
        axs[i, 1].set_xlabel("Step")
        axs[i, 1].set_ylabel("Y Speed")

        # 第三列: 奖励变化
        axs[i, 2].plot(rewards)
        axs[i, 2].set_title(f"Agent {i+1} - Rewards Change")
        axs[i, 2].set_xlabel("Step")
        axs[i, 2].set_ylabel("Reward")

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 调用函数示例：
# 假设 CSV 文件路径是 "agent_steps.csv"，并且我们想分析 Episode 16
csv_file_path = './savegoodresults/agent_steps.csv'
read_and_plot_csv(csv_file_path, episode_number=72)
