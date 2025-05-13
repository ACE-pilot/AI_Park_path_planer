import os
import time
import argparse
import numpy as np
# from simple_model import MAModel
from cnn_model_fast import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.env.multiagent_env import MAenv
from parl.utils import logger, summary
from gym import spaces
from park_env import ParkEnv
from MADDPGWrapper import MADDPGWrapper
import paddle

import psutil  # 用于资源监控
import pynvml  # 用于GPU资源监控

# 设置设备为 CPU
paddle.set_device('gpu')

CRITIC_LR = 0.01  # critic模型的学习率
ACTOR_LR = 0.01  # actor模型的学习率
GAMMA = 0.95  # 奖励折扣因子
TAU = 0.01  # soft update参数
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 30  # 每个回合的最大步数
EVAL_EPISODES = 3  # 评估回合数


def log_resource_usage():
    """
    记录当前的CPU、内存和GPU使用情况。
    """
    # CPU和内存使用情况
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    logger.debug(f"CPU 使用率: {cpu_percent}%, 内存使用率: {memory_info.percent}%")
    
    # GPU使用情况（如果使用GPU）
    if paddle.device.get_device().startswith('gpu'):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            logger.debug(f"GPU 内存使用: {gpu_mem.used / (1024 ** 2)} MB, GPU 利用率: {gpu_util.gpu}%")
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"无法记录GPU使用情况: {e}")


def run_evaluate_episodes(env, agents, eval_episodes):
    """
    运行评估回合，返回奖励和步数。
    """
    logger.info(f"开始运行 {eval_episodes} 个评估回合。")
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        logger.debug("评估: 环境已重置。")
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [
                agent.predict(obs) for agent, obs in zip(agents, obs_n)
            ]
            logger.debug(f"评估回合步骤 {steps}: 预测的动作: {action_n}")
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            total_reward += sum(reward_n)
            logger.debug(f"评估回合步骤 {steps}: 奖励: {reward_n}, 完成标志: {done_n}")

            # 显示动画
            if args.show:
                time.sleep(0.1)
                env.render()

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
        logger.debug(f"评估回合 {len(eval_episode_rewards)} 完成: 奖励 {total_reward}, 步数 {steps}")
    logger.info("评估完成。")
    return eval_episode_rewards, eval_episode_steps


def run_episode(env, agents):
    """
    运行一个训练回合，返回总奖励、每个智能体的奖励和步数。
    """
    logger.debug("开始一个新的训练回合。")
    obs_n = env.reset()
    logger.debug("环境已重置。")
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        logger.debug(f"训练回合步骤 {steps}: 采样的动作: {action_n}")
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        logger.debug(f"训练回合步骤 {steps}: 奖励: {reward_n}, 完成标志: {done_n}")

        # 存储经验
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i], next_obs_n[i], done_n[i])
            logger.debug(f"智能体 {i}: 已添加经验。")

        # 计算每个智能体的奖励
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # 如果是恢复模式并且需要显示，则跳过学习步骤
        if args.restore and args.show:
            logger.debug("恢复模式并且启用了显示，跳过学习步骤。")
            continue

        # 学习策略
        for i, agent in enumerate(agents):
            try:
                critic_loss = agent.learn(agents)
                logger.debug(f"智能体 {i}: Critic 损失: {critic_loss}")
            except Exception as e:
                logger.error(f"智能体 {i}: 学习失败，错误: {e}")
                raise e

    logger.debug(f"训练回合结束: 总奖励: {total_reward}, 每个智能体的奖励: {agents_reward}, 步数: {steps}")
    return total_reward, agents_reward, steps


def main():
    try:
        # 设置日志目录和级别
        logger.set_dir('./train_log/{}_{}'.format(args.env, args.continuous_actions))
        logger.set_level('DEBUG')  # 设置为DEBUG级别以获取详细日志
        logger.info("训练开始。")

        # 创建环境实例
        env = ParkEnv(num_agents=3, render_mode=False)
        wrapped_env = MADDPGWrapper(env, continuous_actions=True)
        env = wrapped_env
        logger.info("环境已创建并包装。")

        # 打印 action_space 的类型和内容
        print(f"Type of env.action_space: {type(env.action_space)}")
        print(f"env.action_space keys: {list(env.action_space.keys())}")
        logger.info(f"动作空间: {env.action_space}")

        if args.continuous_actions:
            assert isinstance(env.action_space['agent_0'], spaces.Box), "动作空间应为 Box 类型以支持连续动作"
            logger.info("使用连续动作空间。")

        # 计算观测空间展平后的尺寸
        obs_size_n = [int(np.prod(shape)) for shape in env.obs_shape_n]
        act_dim_n = env.act_shape_n  # 例如 [2, 2, 2]
        critic_in_dim = sum(obs_size_n) + sum(act_dim_n)
        logger.info(f"环境观测空间形状: {env.obs_shape_n}, 动作空间维度: {env.act_shape_n}")
        logger.info(f"观测空间展平后的尺寸: {obs_size_n}")
        logger.info(f"Critic 输入维度: {critic_in_dim}")

        # 构建所有智能体的动作空间列表
        act_space_n = [env.action_space[f'agent_{i}'] for i in range(env.n)]
        logger.info("所有智能体的动作空间列表已构建。")

        # 构建智能体
        agents = []
        for i in range(env.n):
            model = MAModel(
                obs_dim=env.obs_shape_n[i],      # 单个智能体的观测空间形状
                act_dim=env.act_shape_n[i],      # 单个智能体的动作空间维度
                obs_shape_n=env.obs_shape_n,     # 所有智能体的观测空间形状列表
                act_shape_n=env.act_shape_n,     # 所有智能体的动作空间维度列表
                continuous_actions=args.continuous_actions
            )
            algorithm = MADDPG(
                model,
                agent_index=i,
                act_space=act_space_n,  # 传递所有智能体的动作空间列表
                gamma=GAMMA,
                tau=TAU,
                critic_lr=CRITIC_LR,
                actor_lr=ACTOR_LR
            )
            agent = MAAgent(
                algorithm,
                agent_index=i,
                obs_dim_n=env.obs_shape_n,  # 包含所有智能体观测空间形状的列表
                act_dim_n=env.act_shape_n,  # 包含所有智能体动作空间维度的列表
                batch_size=BATCH_SIZE
            )
            agents.append(agent)
            logger.info(f"智能体 {i} 已创建并添加到智能体列表。")

        if args.restore:
            # 恢复模型
            for i in range(len(agents)):
                model_file = os.path.join(args.model_dir, f'agent_{i}')
                if not os.path.exists(model_file):
                    logger.error(f"模型文件 {model_file} 不存在。")
                    raise Exception(f'模型文件 {model_file} 不存在')
                agents[i].restore(model_file)
                logger.info(f"智能体 {i} 的模型已从 {model_file} 恢复。")

        total_steps = 0
        total_episodes = 0
        logger.info("开始训练循环。")
        while total_episodes <= args.max_episodes:
            # 运行一个回合
            ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
            logger.debug(f"回合 {total_episodes} 完成，奖励: {ep_reward}, 步数: {steps}。")

            # 在每个回合结束后，如果需要，渲染环境
            if args.show:
                env.render()

            # 记录和打印信息        
            summary.add_scalar('train/episode_reward_wrt_episode', ep_reward, total_episodes)
            summary.add_scalar('train/episode_reward_wrt_step', ep_reward, total_steps)
            logger.info(
                f'总步数: {total_steps}, 回合: {total_episodes}, 奖励: {ep_reward}, 智能体奖励: {ep_agent_rewards}, 回合步数: {steps}'
            )

            total_steps += steps
            total_episodes += 1

            # 记录资源使用情况
            log_resource_usage()

            # 评估智能体
            if total_episodes % args.test_every_episodes == 0:
                logger.info(f"在回合 {total_episodes} 开始评估。")
                eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(env, agents, EVAL_EPISODES)
                summary.add_scalar('eval/episode_reward', np.mean(eval_episode_rewards), total_episodes)
                logger.info(f'评估完成: {EVAL_EPISODES} 个回合, 平均奖励: {np.mean(eval_episode_rewards)}')

                # 保存模型
                if not args.restore:
                    model_dir = args.model_dir
                    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                    for i in range(len(agents)):
                        model_name = f'/agent_{i}'
                        agents[i].save(model_dir + model_name)
                        logger.info(f"智能体 {i} 的模型已保存到 {model_dir + model_name}。")

        logger.info("训练完成。")
    except Exception as e:
        logger.error(f"发生未预期的错误: {e}", exc_info=True)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境相关参数
    parser.add_argument(
        '--env',
        type=str,
        default='Park_Env',
        help='Park_Env 由 Zihuan Zhang 设计')
    # 自动保存模型，可选恢复模型
    parser.add_argument(
        '--show', action='store_true', default=False, help='是否显示训练过程')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='是否恢复训练，需指定 model_dir')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='保存模型的目录')
    parser.add_argument(
        '--continuous_actions',
        action='store_true',
        default=True,
        help='是否使用连续动作模式')
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=25000,
        help='停止条件: 回合数')
    parser.add_argument(
        '--test_every_episodes',
        type=int,
        default=int(1e3),
        help='每隔多少回合进行一次评估')
    args = parser.parse_args()

    main()
