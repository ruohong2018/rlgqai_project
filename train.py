"""
Main Training Script for RLGQAI
RLGQAI主训练脚本
"""

import argparse
import os
import sys
import numpy as np
import torch
import random
from datetime import datetime

from qc_maddpg import QCMADDPG
from config import Config


def set_random_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RLGQAI Training Script')
    
    # 训练参数
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto)')
    
    # 模型参数
    parser.add_argument('--model-type', type=str, default='QVGAN',
                       choices=['QVGAN', 'QBM', 'QVAE'],
                       help='Quantum AI model type (default: QVGAN)')
    parser.add_argument('--num-qubits', type=int, default=4,
                       help='Number of qubits (default: 4)')
    
    # 学习参数
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                       help='Actor learning rate (default: 1e-4)')
    parser.add_argument('--critic-lr', type=float, default=3e-4,
                       help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--buffer-size', type=int, default=10000,
                       help='Replay buffer size (default: 10000)')
    
    # 保存和日志
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save frequency in episodes (default: 50)')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Log frequency in episodes (default: 10)')
    
    # 其他
    parser.add_argument('--load-checkpoint', type=str, default=None,
                       help='Path to load checkpoint from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only (no training)')
    
    args = parser.parse_args()
    return args


def setup_directories(args):
    """创建必要的目录"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Log directory: {args.log_dir}")


def create_config(args):
    """根据命令行参数创建配置"""
    cfg = Config()
    
    # 更新配置
    cfg.NUM_EPISODES = args.episodes
    cfg.RANDOM_SEED = args.seed
    cfg.ACTOR_LR = args.actor_lr
    cfg.CRITIC_LR = args.critic_lr
    cfg.BATCH_SIZE = args.batch_size
    cfg.BUFFER_SIZE = args.buffer_size
    cfg.CHECKPOINT_DIR = args.checkpoint_dir
    cfg.LOG_DIR = args.log_dir
    cfg.SAVE_FREQUENCY = args.save_freq
    cfg.LOG_FREQUENCY = args.log_freq
    
    return cfg


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # 创建目录
    setup_directories(args)
    
    # 创建配置
    cfg = create_config(args)
    
    # 打印配置
    print("\n" + "="*60)
    print("RLGQAI Configuration")
    print("="*60)
    print(f"Model Type: {args.model_type}")
    print(f"Number of Qubits: {args.num_qubits}")
    print(f"Training Episodes: {args.episodes}")
    print(f"Actor Learning Rate: {args.actor_lr}")
    print(f"Critic Learning Rate: {args.critic_lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Buffer Size: {args.buffer_size}")
    print("="*60 + "\n")
    
    # 创建算法实例
    print("Initializing QC-MADDPG...")
    algorithm = QCMADDPG(cfg=cfg)
    
    # 加载检查点（如果指定）
    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        algorithm.load_models(args.load_checkpoint)
    
    # 评估模式
    if args.eval_only:
        print("\nRunning evaluation only...")
        # 这里可以添加评估代码
        print("Evaluation completed!")
        return
    
    # 训练
    start_time = datetime.now()
    print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        algorithm.train(num_episodes=args.episodes)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        save_choice = input("Save current models? (y/n): ")
        if save_choice.lower() == 'y':
            algorithm.save_models("interrupted")
            print("Models saved!")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 训练结束
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {duration}")
    
    # 保存最终模型
    print("\nSaving final models...")
    algorithm.save_models("final")
    
    print("\n" + "="*60)
    print("Training Successfully Completed!")
    print("="*60)


if __name__ == "__main__":
    main()

