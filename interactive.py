import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
from park_envE import ParkEnv

# 初始化环境
env = ParkEnv(num_agents=3, render_mode=False)
agents = env.agents  # 使用三个智能体进行测试

print("Initializing environment...")
env.reset()
print("Environment initialized.")

# 设置PyGame
scaling_factor = 10  # 缩放因子
pygame.init()
size = (env.map_size * scaling_factor, env.map_size * scaling_factor + 100)  # 每个单元格放大为 scaling_factor x scaling_factor 像素，增加 100 像素用于显示分数标签
screen = pygame.display.set_mode(size)
pygame.display.set_caption("MADDPG Environment Test")
clock = pygame.time.Clock()
frame_rate = 30  # 帧率配置参数

font = pygame.font.SysFont("Arial", 20)

# 定义每个智能体的颜色
agent_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红色、绿色、蓝色

# 绘制地图
def draw_map():
    screen.fill((255, 255, 255))

    # 显示高差信息（首先绘制高差，确保其他内容不被覆盖）
    elevation_map = env.map[1]
    elevation_surface = pygame.Surface((env.map_size * scaling_factor, env.map_size * scaling_factor), pygame.SRCALPHA)
    for y in range(env.map_size):
        for x in range(env.map_size):
            elevation_value = elevation_map[y, x]
            if elevation_value >= 0:
                color = plt.cm.terrain(elevation_value / 255.0)  # 使用 terrain 颜色映射获取颜色
                color = tuple(int(c * 255) for c in color[:3]) + (int(0.8 * 255),)  # 转换为 RGBA，设置透明度为 0.8
                pygame.draw.rect(elevation_surface, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))
    screen.blit(elevation_surface, (0, 0))

    # 绘制等高线
    contour_levels = np.linspace(np.min(elevation_map), np.max(elevation_map), 10)
    fig, ax = plt.subplots()
    ax.contour(elevation_map, levels=contour_levels, cmap='terrain', alpha=0.5)
    plt.close(fig)

    # 绘制其他元素
    for y in range(env.map_size):
        for x in range(env.map_size):
            if env.map[2, y, x] >= 100:
                color = (0, 255, 0)  # 轨迹为绿色
                pygame.draw.rect(screen, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))
            elif env.map[2, y, x] == env.START_POINT_VALUE:
                color = (255, 255, 0)  # 起点为黄色
                pygame.draw.rect(screen, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))
            elif env.map[2, y, x] == 1:
                color = (34, 139, 34)  # 树木为森林绿色
                pygame.draw.rect(screen, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))
            elif env.map[2, y, x] == 2:
                color = (160, 82, 45)  # 构筑物为棕色
                pygame.draw.rect(screen, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))
            elif env.map[2, y, x] == 3:
                color = (0, 255, 255)  # 人工景观为青色
                pygame.draw.rect(screen, color, (x * scaling_factor, y * scaling_factor, scaling_factor, scaling_factor))

    # 绘制边界为红色线条（使用生成边界的角点连线）
    if hasattr(env, 'boundary_polygon') and env.boundary_polygon is not None:
        boundary_coords = list(env.boundary_polygon.exterior.coords)
        scaled_boundary = [(x * scaling_factor, y * scaling_factor) for x, y in boundary_coords]
        pygame.draw.lines(screen, (255, 0, 0), True, scaled_boundary, 2)  # 使用红色线条绘制边界线

    # 绘制终点为红色五角星
    if hasattr(env, 'end_pos') and env.end_pos is not None:
        end_x, end_y = env.end_pos
        end_pixel_pos = (int(end_x * scaling_factor), int(end_y * scaling_factor))
        pygame.draw.polygon(screen, (255, 0, 0), [(end_pixel_pos[0], end_pixel_pos[1] - scaling_factor),
                                                  (end_pixel_pos[0] + 4, end_pixel_pos[1] - 3),
                                                  (end_pixel_pos[0] + scaling_factor, end_pixel_pos[1] - 3),
                                                  (end_pixel_pos[0] + 6, end_pixel_pos[1] + 2),
                                                  (end_pixel_pos[0] + 8, end_pixel_pos[1] + scaling_factor),
                                                  (end_pixel_pos[0], end_pixel_pos[1] + 5),
                                                  (end_pixel_pos[0] - 8, end_pixel_pos[1] + scaling_factor),
                                                  (end_pixel_pos[0] - 6, end_pixel_pos[1] + 2),
                                                  (end_pixel_pos[0] - scaling_factor, end_pixel_pos[1] - 3),
                                                  (end_pixel_pos[0] - 4, end_pixel_pos[1] - 3)])

    # 绘制每个智能体的轨迹
    for i, agent in enumerate(agents):
        trajectory = env.trajectories[i]
        if len(trajectory) > 1:
            scaled_trajectory = [(pos[0] * scaling_factor, pos[1] * scaling_factor) for pos in trajectory]
            pygame.draw.lines(screen, agent_colors[i], False, scaled_trajectory, 2)  # 使用相应颜色绘制智能体轨迹

# 鼠标点击位置换算为智能体动作
def get_action_from_click(click_pos, agent_pos):
    action = np.zeros(4)
    dx = (click_pos[0] - agent_pos[0])
    dy = (click_pos[1] - agent_pos[1])
    

    # 标准化 dx 和 dy 到 -1 到 1 的范围
    distance = np.sqrt(dx**2 + dy**2)
    if distance > 1:
        dx /= distance
        dy /= distance
    
    # 计算水平方向的动作
    if dx > 0:
        action[1] = abs(dx)  # 向右
    elif dx < 0:
        action[0] = abs(dx)  # 向左
    else:
        action[0] = 0
        action[1] = 0

    # 计算垂直方向的动作
    if dy > 0:
        action[3] = abs(dy)  # 向下
    elif dy < 0:
        action[2] = abs(dy)  # 向上
    else:
        action[2] = 0
        action[3] = 0

    print(f"Click position: {click_pos}, Agent position: {agent_pos}, dx: {dx}, dy: {dy}")
    print(f"Calculated action: {action}")
    return action

def main():
    running = True
    env.reset()
    print("Environment reset.")
    total_score = [0, 0, 0]
    step = 0
    episode = 1
    clicks = 0
    actions = {}

    while running:
        draw_map()
        for i, agent in enumerate(agents):
            agent_pos = env.agent_positions[i]
            pygame.draw.circle(screen, agent_colors[i], (int(agent_pos[0] * scaling_factor), int(agent_pos[1] * scaling_factor)), scaling_factor)  # 使用不同颜色的圆点区分智能体
            agent_label = font.render(str(i + 1), True, (0, 0, 0))  # 绘制智能体的序号
            screen.blit(agent_label, (int(agent_pos[0] * scaling_factor) - scaling_factor // 2, int(agent_pos[1] * scaling_factor) - scaling_factor // 2))

        # 显示每个智能体的得分
        for i, agent in enumerate(agents):
            score_text = f"Agent {i + 1} Score: {total_score[i]}"
            score_surface = font.render(score_text, True, (0, 0, 0))
            screen.blit(score_surface, (10, env.map_size * scaling_factor + 20 * i))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                click_pos = pygame.mouse.get_pos()
                click_pos = (click_pos[0] / scaling_factor, click_pos[1] / scaling_factor)
                agent_index = clicks % 3
                print(f"Mouse clicked at: {click_pos} for agent {agent_index}")
                action = get_action_from_click(click_pos, env.agent_positions[agent_index])
                actions[agents[agent_index]] = action
                clicks += 1

                if clicks % 3 == 0:
                    try:
                        obs, rewards, dones, infos = env.step(actions)
                        #print(f"Observations: {obs}, Rewards: {rewards}, Dones: {dones}, Infos: {infos}")
                        for i, agent in enumerate(agents):
                            total_score[i] += rewards[agent]
                            print(f"Score for agent {agent}: {rewards[agent]}")
                        step += 1
                        print(f"Step: {step}, Total Score: {total_score}")

                        if all(dones.values()):
                            print(f"Episode {episode} finished in {step} steps with total scores: {total_score}")
                            time.sleep(2)
                            env.reset()
                            print(f"Environment reset for new episode {episode + 1}.")
                            episode += 1
                            print(f"Total score for episode {episode - 1}: {total_score}")
                            total_score = [0, 0, 0]
                            step = 0
                    except ZeroDivisionError:
                        print("ZeroDivisionError encountered. Skipping this step.")

        clock.tick(frame_rate)

    pygame.quit()
    print("Game quit.")

if __name__ == "__main__":
    main()
