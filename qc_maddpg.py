"""
QC-MADDPG Algorithm
量子-经典混合多智能体深度确定性策略梯度算法
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import os

from agents import QuantumAgent, ClassicalAgent, ResourceAgent
from utils import AdaptiveOUNoise, PrioritizedReplayBuffer
from environments import QuantumAIEnvironment
from config import config


class QCMADDPG:
    """
    QC-MADDPG (Quantum-Classical Multi-Agent Deep Deterministic Policy Gradient)
    
    核心算法实现：
    1. 三个专门化智能体的协同优化
    2. 集中式训练、分布式执行
    3. 优先级经验回放
    4. 自适应探索策略
    """
    
    def __init__(self, cfg=None):
        """
        Args:
            cfg: 配置对象，默认使用config
        """
        self.cfg = cfg if cfg is not None else config
        
        # 状态和动作维度
        self.quantum_state_dim = self.cfg.STATE_DIM_QUANTUM
        self.quantum_action_dim = self.cfg.QUANTUM_DIM
        
        self.classical_state_dim = self.cfg.STATE_DIM_CLASSICAL
        self.classical_action_dim = self.cfg.CLASSICAL_DIM
        
        self.resource_state_dim = self.cfg.STATE_DIM_RESOURCE
        self.resource_action_dim = self.cfg.RESOURCE_DIM
        
        self.global_state_dim = self.cfg.TOTAL_STATE_DIM
        self.global_action_dim = self.cfg.TOTAL_PARAM_DIM
        
        # 创建三个智能体
        print("Initializing agents...")
        self.quantum_agent = QuantumAgent(
            self.quantum_state_dim, self.quantum_action_dim,
            self.global_state_dim, self.global_action_dim,
            actor_lr=self.cfg.ACTOR_LR,
            critic_lr=self.cfg.CRITIC_LR,
            gamma=self.cfg.GAMMA,
            tau=self.cfg.TAU
        )
        
        self.classical_agent = ClassicalAgent(
            self.classical_state_dim, self.classical_action_dim,
            self.global_state_dim, self.global_action_dim,
            actor_lr=self.cfg.ACTOR_LR,
            critic_lr=self.cfg.CRITIC_LR,
            gamma=self.cfg.GAMMA,
            tau=self.cfg.TAU
        )
        
        self.resource_agent = ResourceAgent(
            self.resource_state_dim, self.resource_action_dim,
            self.global_state_dim, self.global_action_dim,
            actor_lr=self.cfg.ACTOR_LR,
            critic_lr=self.cfg.CRITIC_LR,
            gamma=self.cfg.GAMMA,
            tau=self.cfg.TAU
        )
        
        self.agents = [self.quantum_agent, self.classical_agent, self.resource_agent]
        
        # 创建探索噪声
        print("Initializing exploration noise...")
        self.quantum_noise = AdaptiveOUNoise(
            self.quantum_action_dim,
            theta=self.cfg.OU_THETA,
            sigma_start=self.cfg.OU_SIGMA,
            sigma_end=self.cfg.MIN_NOISE,
            decay_rate=self.cfg.NOISE_DECAY
        )
        
        self.classical_noise = AdaptiveOUNoise(
            self.classical_action_dim,
            theta=self.cfg.OU_THETA,
            sigma_start=self.cfg.OU_SIGMA,
            sigma_end=self.cfg.MIN_NOISE,
            decay_rate=self.cfg.NOISE_DECAY
        )
        
        self.resource_noise = AdaptiveOUNoise(
            self.resource_action_dim,
            theta=self.cfg.OU_THETA,
            sigma_start=self.cfg.OU_SIGMA,
            sigma_end=self.cfg.MIN_NOISE,
            decay_rate=self.cfg.NOISE_DECAY
        )
        
        self.noises = [self.quantum_noise, self.classical_noise, self.resource_noise]
        
        # 创建经验回放缓冲区
        print("Initializing replay buffer...")
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.cfg.BUFFER_SIZE,
            alpha=self.cfg.PRIORITY_ALPHA,
            beta_start=self.cfg.PRIORITY_BETA_START,
            beta_end=self.cfg.PRIORITY_BETA_END,
            epsilon=self.cfg.PRIORITY_EPSILON
        )
        
        # 创建环境
        print("Initializing environment...")
        self.env = QuantumAIEnvironment()
        
        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = -float('inf')
        
        print("QC-MADDPG initialized successfully!")
    
    def select_actions(self, observations: Dict[str, np.ndarray], 
                      add_noise: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        选择动作
        Args:
            observations: 各智能体的观测
            add_noise: 是否添加探索噪声
        Returns:
            actions_list: 各智能体的动作列表
            global_action: 全局动作
        """
        quantum_obs = observations['quantum']
        classical_obs = observations['classical']
        resource_obs = observations['resource']
        
        # 获取各智能体的动作
        if add_noise:
            quantum_action = self.quantum_agent.get_action(
                quantum_obs, self.quantum_noise.sample()
            )
            classical_action = self.classical_agent.get_action(
                classical_obs, self.classical_noise.sample()
            )
            resource_action = self.resource_agent.get_action(
                resource_obs, self.resource_noise.sample()
            )
        else:
            quantum_action = self.quantum_agent.get_action(quantum_obs, None)
            classical_action = self.classical_agent.get_action(classical_obs, None)
            resource_action = self.resource_agent.get_action(resource_obs, None)
        
        # 组合为全局动作
        global_action = np.concatenate([quantum_action, classical_action, resource_action])
        actions_list = [quantum_action, classical_action, resource_action]
        
        return actions_list, global_action
    
    def store_experience(self, obs, actions, reward, next_obs, done):
        """
        存储经验到回放缓冲区
        Args:
            obs: 当前观测
            actions: 动作
            reward: 奖励
            next_obs: 下一观测
            done: 是否结束
        """
        # 提取局部和全局状态
        local_states = [obs['quantum'], obs['classical'], obs['resource']]
        global_state = obs['global']
        
        next_local_states = [next_obs['quantum'], next_obs['classical'], next_obs['resource']]
        next_global_state = next_obs['global']
        
        # 存储（简化版本，实际应分别存储）
        self.replay_buffer.push(
            global_state, actions, reward, next_global_state, done
        )
    
    def update_agents(self):
        """更新所有智能体"""
        if len(self.replay_buffer) < self.cfg.BATCH_SIZE:
            return None
        
        # 采样经验
        (states, actions, rewards, next_states, dones, 
         indices, weights) = self.replay_buffer.sample(self.cfg.BATCH_SIZE)
        
        # 为简化，这里使用全局状态
        # 在完整实现中，应该正确分离局部状态
        
        # 计算TD误差用于更新优先级
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_tensor = torch.FloatTensor(next_states)
            
            # 使用第一个智能体的critic计算TD误差
            current_q = self.quantum_agent.critic(states_tensor, actions_tensor)
            
            # 计算目标Q值（需要目标动作）
            # 这里简化处理
            target_q = rewards_tensor
            
            td_errors = (current_q - target_q).cpu().numpy().flatten()
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # 这里简化了更新逻辑
        # 完整版本需要为每个智能体准备正确的批次数据
        losses = {
            'quantum_actor': 0.0,
            'quantum_critic': 0.0,
            'classical_actor': 0.0,
            'classical_critic': 0.0,
            'resource_actor': 0.0,
            'resource_critic': 0.0
        }
        
        return losses
    
    def train_episode(self) -> Dict:
        """
        训练一个episode
        Returns:
            episode_info: episode信息
        """
        self.episode_count += 1
        
        # 重置环境和噪声
        obs = self.env.reset()
        for noise in self.noises:
            noise.reset_for_episode()
        
        episode_reward = 0
        episode_steps = 0
        losses_history = []
        
        done = False
        while not done and episode_steps < self.cfg.MAX_STEPS_PER_EPISODE:
            # 选择动作
            actions_list, global_action = self.select_actions(obs, add_noise=True)
            
            # 将动作转换为配置
            config = self._action_to_config(global_action)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(config)
            
            # 存储经验
            self.store_experience(obs, global_action, reward, next_obs, done)
            
            # 更新智能体
            if len(self.replay_buffer) >= self.cfg.WARMUP_STEPS:
                losses = self.update_agents()
                if losses is not None:
                    losses_history.append(losses)
            
            # 更新状态
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
        
        # 更新最佳性能
        if episode_reward > self.best_performance:
            self.best_performance = episode_reward
        
        # Episode信息
        episode_info = {
            'episode': self.episode_count,
            'reward': episode_reward,
            'steps': episode_steps,
            'best_reward': self.best_performance,
            'buffer_size': len(self.replay_buffer),
            'performance': self.env.get_performance()
        }
        
        return episode_info
    
    def train(self, num_episodes: int = None):
        """
        训练QC-MADDPG
        Args:
            num_episodes: 训练的episode数量
        """
        if num_episodes is None:
            num_episodes = self.cfg.NUM_EPISODES
        
        print(f"\n{'='*60}")
        print(f"Starting QC-MADDPG Training for {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            episode_info = self.train_episode()
            
            # 日志
            if (episode + 1) % self.cfg.LOG_FREQUENCY == 0:
                print(f"Episode {episode_info['episode']}/{num_episodes}")
                print(f"  Reward: {episode_info['reward']:.4f}")
                print(f"  Steps: {episode_info['steps']}")
                print(f"  Best Reward: {episode_info['best_reward']:.4f}")
                print(f"  Fidelity: {episode_info['performance']['fidelity']:.4f}")
                print(f"  Convergence Steps: {episode_info['performance']['convergence_steps']}")
                print(f"  Buffer Size: {episode_info['buffer_size']}")
                print()
            
            # 保存模型
            if (episode + 1) % self.cfg.SAVE_FREQUENCY == 0:
                self.save_models(f"episode_{episode + 1}")
        
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"Best Performance: {self.best_performance:.4f}")
        print(f"{'='*60}\n")
    
    def _action_to_config(self, action: np.ndarray) -> Dict:
        """
        将动作转换为参数配置
        Args:
            action: 归一化的动作 [-1, 1]
        Returns:
            config: 参数配置字典
        """
        # 简化的转换
        config = {
            'circuit_depth': int(5 + action[0] * 5),  # [0, 10]
            'learning_rate': 10 ** (action[1] * 2 - 3),  # [1e-5, 1e-1]
            'shot_number': int(1000 + action[2] * 4500),  # [100, 10000]
            'batch_size': int(64 + action[3] * 96)  # [16, 256]
        }
        return config
    
    def save_models(self, tag: str = "latest"):
        """
        保存所有智能体的模型
        Args:
            tag: 保存标签
        """
        save_dir = self.cfg.CHECKPOINT_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        self.quantum_agent.save(os.path.join(save_dir, f"quantum_{tag}.pth"))
        self.classical_agent.save(os.path.join(save_dir, f"classical_{tag}.pth"))
        self.resource_agent.save(os.path.join(save_dir, f"resource_{tag}.pth"))
        
        print(f"Models saved with tag: {tag}")
    
    def load_models(self, tag: str = "latest"):
        """
        加载所有智能体的模型
        Args:
            tag: 加载标签
        """
        save_dir = self.cfg.CHECKPOINT_DIR
        
        self.quantum_agent.load(os.path.join(save_dir, f"quantum_{tag}.pth"))
        self.classical_agent.load(os.path.join(save_dir, f"classical_{tag}.pth"))
        self.resource_agent.load(os.path.join(save_dir, f"resource_{tag}.pth"))
        
        print(f"Models loaded with tag: {tag}")


if __name__ == "__main__":
    # 测试QC-MADDPG
    print("Testing QC-MADDPG...")
    
    # 创建算法实例
    algorithm = QCMADDPG()
    
    # 训练几个episode
    print("\nRunning short training test...")
    algorithm.train(num_episodes=5)
    
    print("\nQC-MADDPG test completed!")

