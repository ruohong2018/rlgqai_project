"""
Experience Replay Buffer with Prioritized Sampling
优先级经验回放缓冲区
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    标准经验回放缓冲区
    存储和采样经验元组 (state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 缓冲区最大容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        Args:
            batch_size: 批量大小
        Returns:
            states, actions, rewards, next_states, dones: 经验批次
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    基于TD误差为经验分配优先级，高优先级经验被更频繁采样
    
    优先级计算: priority = |TD_error| + ε
    采样概率: P(i) = priority_i^α / Σ_k priority_k^α
    重要性采样权重: w_i = (N * P(i))^(-β)
    """
    
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_end=1.0, epsilon=1e-6):
        """
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数，控制优先级影响程度
            beta_start: 重要性采样权重初始值
            beta_end: 重要性采样权重最终值
            epsilon: 防止零优先级的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # 用于计算beta的衰减
        self.frame_count = 0
        self.max_frames = 1000000  # 总训练步数
    
    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区，初始优先级设为最大值
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        按优先级采样一批经验
        Args:
            batch_size: 批量大小
        Returns:
            经验批次、采样索引、重要性采样权重
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 按概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 获取经验
        batch = [self.buffer[idx] for idx in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # 计算重要性采样权重
        beta = self._get_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化
        
        self.frame_count += 1
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        更新优先级
        Args:
            indices: 经验索引
            td_errors: TD误差
        """
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
    
    def _get_beta(self):
        """
        计算当前的beta值（线性退火）
        """
        progress = min(self.frame_count / self.max_frames, 1.0)
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        return beta
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return self.size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0


class MultiAgentReplayBuffer:
    """
    多智能体经验回放缓冲区
    存储包含多个智能体观测和动作的经验
    """
    
    def __init__(self, capacity=10000, num_agents=3):
        """
        Args:
            capacity: 缓冲区最大容量
            num_agents: 智能体数量
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(self, observations, actions, reward, next_observations, done):
        """
        添加多智能体经验
        Args:
            observations: 各智能体的观测列表
            actions: 各智能体的动作列表
            reward: 共享奖励
            next_observations: 各智能体的下一观测列表
            done: 是否结束
        """
        self.buffer.append((observations, actions, reward, next_observations, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        Args:
            batch_size: 批量大小
        Returns:
            各智能体的观测、动作、奖励、下一观测、结束标志
        """
        batch = random.sample(self.buffer, batch_size)
        
        # 分离各智能体的数据
        observations = [[] for _ in range(self.num_agents)]
        actions = [[] for _ in range(self.num_agents)]
        rewards = []
        next_observations = [[] for _ in range(self.num_agents)]
        dones = []
        
        for exp in batch:
            obs, acts, rew, next_obs, done = exp
            
            for i in range(self.num_agents):
                observations[i].append(obs[i])
                actions[i].append(acts[i])
                next_observations[i].append(next_obs[i])
            
            rewards.append(rew)
            dones.append(done)
        
        # 转换为numpy数组
        observations = [np.array(obs) for obs in observations]
        actions = [np.array(acts) for acts in actions]
        rewards = np.array(rewards)
        next_observations = [np.array(next_obs) for next_obs in next_observations]
        dones = np.array(dones)
        
        return observations, actions, rewards, next_observations, dones
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


if __name__ == "__main__":
    # 测试代码
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    # 添加一些经验
    for i in range(50):
        state = np.random.randn(10)
        action = np.random.randn(5)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # 采样
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Sampled batch shapes: states={states.shape}, actions={actions.shape}")
    
    # 测试优先级回放
    print("\nTesting PrioritizedReplayBuffer...")
    
    priority_buffer = PrioritizedReplayBuffer(capacity=100)
    
    # 添加经验
    for i in range(50):
        state = np.random.randn(10)
        action = np.random.randn(5)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        priority_buffer.push(state, action, reward, next_state, done)
    
    print(f"Priority buffer size: {len(priority_buffer)}")
    
    # 采样
    states, actions, rewards, next_states, dones, indices, weights = priority_buffer.sample(32)
    print(f"Sampled with priorities - weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # 更新优先级
    td_errors = np.random.randn(32)
    priority_buffer.update_priorities(indices, td_errors)
    print("Priorities updated")
    
    # 测试多智能体缓冲区
    print("\nTesting MultiAgentReplayBuffer...")
    
    ma_buffer = MultiAgentReplayBuffer(capacity=100, num_agents=3)
    
    # 添加经验
    for i in range(50):
        observations = [np.random.randn(10) for _ in range(3)]
        actions = [np.random.randn(5) for _ in range(3)]
        reward = np.random.randn()
        next_observations = [np.random.randn(10) for _ in range(3)]
        done = False
        ma_buffer.push(observations, actions, reward, next_observations, done)
    
    print(f"Multi-agent buffer size: {len(ma_buffer)}")
    
    # 采样
    obs, acts, rews, next_obs, dones = ma_buffer.sample(32)
    print(f"Sampled multi-agent batch:")
    print(f"  Agent 0 observations shape: {obs[0].shape}")
    print(f"  Agent 1 actions shape: {acts[1].shape}")
    print(f"  Rewards shape: {rews.shape}")
    
    print("\nAll tests passed!")

