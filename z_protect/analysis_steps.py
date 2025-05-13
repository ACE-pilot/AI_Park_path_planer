import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_plot_all_episodes(csv_file_path, save_path="all_episodes.svg"):
    df = pd.read_csv(csv_file_path)
    episodes = df['episode'].unique()
    
    # viridis 色图
    cmap = plt.cm.viridis
    color_list = [cmap(i / len(episodes)) for i in range(len(episodes))]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("All Episodes Analysis", fontsize=20)

    column_titles = ["X Speed Change", "Y Speed Change", "Cumulative Reward"]
    for ax, col_title in zip(axs[0], column_titles):
        ax.set_title(col_title, fontsize=12, pad=30)

    legend_lines = []
    legend_labels = []

    for ep_idx, episode_number in enumerate(episodes):
        episode_data = df[df['episode'] == episode_number]
        total_score = 0

        for i in range(3):
            actions = episode_data[f"agent_{i}_actions"].values[0].split(";")
            rewards = episode_data[f"agent_{i}_rewards"].values[0].split(";")

            x_speed = []
            y_speed = []
            for action in actions:
                action_values = action.split(",")
                x_speed.append(float(action_values[1]) - float(action_values[0]))
                y_speed.append(float(action_values[3]) - float(action_values[2]))

            rewards = np.array([float(r) for r in rewards])
            cumulative_rewards = np.cumsum(rewards)
            total_score += cumulative_rewards[-1]

            axs[i, 0].plot(x_speed, color=color_list[ep_idx], alpha=0.8)
            axs[i, 1].plot(y_speed, color=color_list[ep_idx], alpha=0.8)
            axs[i, 2].plot(cumulative_rewards, color=color_list[ep_idx], alpha=0.8)

            if ep_idx == 0:
                axs[i, 0].set_ylabel(f"Agent {i+1}", fontsize=12)

        label_text = f"Episode {episode_number} (Score: {total_score:.1f})"
        legend_lines.append(axs[0, 0].lines[-1])
        legend_labels.append(label_text)

    for j in range(3):
        axs[2, j].set_xlabel("Step", fontsize=12)

    fig.legend(legend_lines, legend_labels,
               loc='lower center', ncol=min(len(legend_labels), 6),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()



# 调用函数
csv_file_path = './savegoodresults/agent_steps.csv'
read_and_plot_all_episodes(csv_file_path, save_path='steps_analysis.svg')
