"""
多种子训练封装。

设置 numpy 和 paddle 的随机种子后，调用 train.py 进行训练。

运行:
    cd examples/MADDPG_18_hyper_vision
    CUDA_VISIBLE_DEVICES=0 python train_with_seed.py --seed 0 --model_dir ./model_seed0
    CUDA_VISIBLE_DEVICES=1 python train_with_seed.py --seed 1 --model_dir ./model_seed1
    CUDA_VISIBLE_DEVICES=2 python train_with_seed.py --seed 2 --model_dir ./model_seed2
"""
import os
import sys
import argparse
import numpy as np
import paddle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='随机种子')
    parser.add_argument('--model_dir', type=str, required=True, help='模型保存目录')
    parser.add_argument('--max_episodes', type=int, default=1000000)
    parser.add_argument('--test_every_episodes', type=int, default=1000)
    args, remaining = parser.parse_known_args()

    # 在其他任何操作之前设置种子
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    train_script = os.path.join(os.path.dirname(__file__), 'train.py')
    train_script = os.path.abspath(train_script)

    cmd_args = [
        sys.executable, train_script,
        '--model_dir', args.model_dir,
        '--max_episodes', str(args.max_episodes),
        '--test_every_episodes', str(args.test_every_episodes),
    ] + remaining

    print(f"Launching: {' '.join(cmd_args)}")
    os.execv(sys.executable, cmd_args)


if __name__ == '__main__':
    main()
