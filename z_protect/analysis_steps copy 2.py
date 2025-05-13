import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_all_episodes(csv_file_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path)

    # 获取所有独立的 episode 数
    episodes = df['episode'].unique()

    # 创建一个 3x3 的图表矩阵
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"All Episodes Analysis", fontsize=16)

    # 遍历每个智能体（3个智能体）
    for i in range(3):
        for episode_number in episodes:
            # 过滤出当前 episode 的数据
            episode_data = df[df['episode'] == episode_number]

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

            # 将奖励转化为浮动值并进行累加
            rewards = np.array([float(r) for r in rewards])
            cumulative_rewards = np.cumsum(rewards)  # 累加奖励值

            # 绘制子图
            # 第一列: X轴速度变化
            axs[i, 0].plot(x_speed, label=f"Episode {episode_number}")
            axs[i, 0].set_title(f"Agent {i+1} - X Speed Change")
            axs[i, 0].set_xlabel("Step")
            axs[i, 0].set_ylabel("X Speed")

            # 第二列: Y轴速度变化
            axs[i, 1].plot(y_speed, label=f"Episode {episode_number}")
            axs[i, 1].set_title(f"Agent {i+1} - Y Speed Change")
            axs[i, 1].set_xlabel("Step")
            axs[i, 1].set_ylabel("Y Speed")

            # 第三列: 累加奖励变化
            axs[i, 2].plot(cumulative_rewards, label=f"Episode {episode_number}")
            axs[i, 2].set_title(f"Agent {i+1} - Cumulative Rewards Change")
            axs[i, 2].set_xlabel("Step")
            axs[i, 2].set_ylabel("Cumulative Reward")

        # 设置每个小图的图例
        axs[i, 0].legend(loc='upper right')
        axs[i, 1].legend(loc='upper right')
        axs[i, 2].legend(loc='upper right')

    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



# 调用函数示例：
# 假设 CSV 文件路径是 "agent_steps.csv"
csv_file_path = './savegoodresults/agent_steps.csv'
read_and_plot_all_episodes(csv_file_path)

