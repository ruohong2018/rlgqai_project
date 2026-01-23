"""
Base Agent Class
智能体基类
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    智能体基类
    定义智能体的基本接口和通用方法
    """
    
    def __init__(self, state_dim, action_dim, agent_name="agent",
                 actor_lr=1e-4, critic_lr=3e-4, gamma=0.99, tau=0.01):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            agent_name: 智能体名称
            actor_lr: 演员网络学习率
            critic_lr: 评论家网络学习率
            gamma: 折扣因子
            tau: 软更新系数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_name = agent_name
        self.gamma = gamma
        self.tau = tau
        
        # 学习率
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # 网络和优化器将在子类中初始化
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        
        self.actor_optimizer = None
        self.critic_optimizer = None
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练统计
        self.actor_loss = 0
        self.critic_loss = 0
    
    @abstractmethod
    def get_action(self, state, noise=None):
        """
        获取动作
        Args:
            state: 当前状态
            noise: 探索噪声
        Returns:
            action: 动作
        """
        pass
    
    @abstractmethod
    def update(self, batch, other_agents_actions=None):
        """
        更新网络
        Args:
            batch: 经验批次
            other_agents_actions: 其他智能体的动作（用于多智能体）
        """
        pass
    
    def soft_update(self, local_model, target_model):
        """
        软更新目标网络
        θ_target = τ * θ_local + (1 - τ) * θ_target
        
        Args:
            local_model: 在线网络
            target_model: 目标网络
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update(self, local_model, target_model):
        """
        硬更新：直接复制权重
        θ_target = θ_local
        
        Args:
            local_model: 在线网络
            target_model: 目标网络
        """
        target_model.load_state_dict(local_model.state_dict())
    
    def save(self, filepath):
        """
        保存模型
        Args:
            filepath: 保存路径
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"[{self.agent_name}] Model saved to {filepath}")
    
    def load(self, filepath):
        """
        加载模型
        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # 同步目标网络
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)
        
        print(f"[{self.agent_name}] Model loaded from {filepath}")
    
    def set_train_mode(self):
        """设置为训练模式"""
        self.actor.train()
        self.critic.train()
    
    def set_eval_mode(self):
        """设置为评估模式"""
        self.actor.eval()
        self.critic.eval()
    
    def to(self, device):
        """移动到指定设备"""
        self.device = device
        if self.actor is not None:
            self.actor.to(device)
            self.actor_target.to(device)
        if self.critic is not None:
            self.critic.to(device)
            self.critic_target.to(device)
    
    def get_actor_loss(self):
        """获取演员损失"""
        return self.actor_loss
    
    def get_critic_loss(self):
        """获取评论家损失"""
        return self.critic_loss
    
    def __str__(self):
        """字符串表示"""
        return f"{self.agent_name}(state_dim={self.state_dim}, action_dim={self.action_dim})"


class DDPGAgent(BaseAgent):
    """
    DDPG智能体实现
    """
    
    def __init__(self, state_dim, action_dim, agent_name="ddpg_agent",
                 actor_lr=1e-4, critic_lr=3e-4, gamma=0.99, tau=0.01,
                 hidden_dims_actor=[256, 256], hidden_dims_critic=[512, 512, 256]):
        """
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            agent_name: 智能体名称
            actor_lr: 演员网络学习率
            critic_lr: 评论家网络学习率
            gamma: 折扣因子
            tau: 软更新系数
            hidden_dims_actor: 演员网络隐藏层维度
            hidden_dims_critic: 评论家网络隐藏层维度
        """
        super().__init__(state_dim, action_dim, agent_name, actor_lr, critic_lr, gamma, tau)
        
        # 导入网络
        from networks.actor_network import ActorNetwork
        from networks.critic_network import CriticNetwork
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims_actor).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dims_actor).to(self.device)
        self.hard_update(self.actor, self.actor_target)
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims_critic).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dims_critic).to(self.device)
        self.hard_update(self.critic, self.critic_target)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def get_action(self, state, noise=None):
        """
        获取动作
        Args:
            state: 当前状态
            noise: 探索噪声
        Returns:
            action: 动作
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        # 添加噪声
        if noise is not None:
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def update(self, batch, other_agents_actions=None):
        """
        更新网络
        Args:
            batch: 经验批次 (states, actions, rewards, next_states, dones)
            other_agents_actions: 其他智能体的动作（DDPG不使用）
        """
        states, actions, rewards, next_states, dones = batch
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        # 记录损失
        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()


if __name__ == "__main__":
    # 测试代码
    print("Testing BaseAgent and DDPGAgent...")
    
    state_dim = 15
    action_dim = 18
    
    # 创建DDPG智能体
    agent = DDPGAgent(state_dim, action_dim, agent_name="test_agent")
    print(f"Agent created: {agent}")
    
    # 测试获取动作
    state = np.random.randn(state_dim)
    action = agent.get_action(state)
    print(f"Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试更新
    batch_size = 32
    states = np.random.randn(batch_size, state_dim)
    actions = np.random.randn(batch_size, action_dim)
    rewards = np.random.randn(batch_size)
    next_states = np.random.randn(batch_size, state_dim)
    dones = np.zeros(batch_size)
    
    batch = (states, actions, rewards, next_states, dones)
    agent.update(batch)
    
    print(f"Actor loss: {agent.get_actor_loss():.4f}")
    print(f"Critic loss: {agent.get_critic_loss():.4f}")
    
    print("\nAll tests passed!")

