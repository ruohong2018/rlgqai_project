"""
Resource Agent
资源智能体：负责资源层参数的优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork


class ResourceAgent(BaseAgent):
    """
    资源智能体
    负责管理和优化资源层参数：
    - 测量次数
    - 编译优化级别
    - 误差缓解方法
    - 并行度
    - 缓存策略等
    """
    
    def __init__(self, state_dim, action_dim, global_state_dim, global_action_dim,
                 agent_name="resource_agent", actor_lr=1e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.01):
        """
        Args:
            state_dim: 局部状态维度（资源层状态）
            action_dim: 动作维度（资源层参数数量）
            global_state_dim: 全局状态维度（所有层状态）
            global_action_dim: 全局动作维度（所有层参数）
            agent_name: 智能体名称
            actor_lr: 演员网络学习率
            critic_lr: 评论家网络学习率
            gamma: 折扣因子
            tau: 软更新系数
        """
        super().__init__(state_dim, action_dim, agent_name, actor_lr, critic_lr, gamma, tau)
        
        self.global_state_dim = global_state_dim
        self.global_action_dim = global_action_dim
        
        # 初始化Actor网络
        self.actor = ActorNetwork(
            state_dim, action_dim,
            hidden_dims=[256, 256],
            use_batch_norm=True
        ).to(self.device)
        self.actor_target = ActorNetwork(
            state_dim, action_dim,
            hidden_dims=[256, 256],
            use_batch_norm=True
        ).to(self.device)
        self.hard_update(self.actor, self.actor_target)
        
        # 初始化Critic网络（接收全局状态和全局动作）
        self.critic = CriticNetwork(
            global_state_dim, global_action_dim,
            hidden_dims=[512, 512, 256],
            use_batch_norm=True
        ).to(self.device)
        self.critic_target = CriticNetwork(
            global_state_dim, global_action_dim,
            hidden_dims=[512, 512, 256],
            use_batch_norm=True
        ).to(self.device)
        self.hard_update(self.critic, self.critic_target)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    def get_action(self, state, noise=None):
        """
        获取资源层参数配置动作
        Args:
            state: 资源层状态观测
            noise: 探索噪声
        Returns:
            action: 资源层参数动作
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        
        # 添加噪声进行探索
        if noise is not None:
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def update(self, batch, all_agents_actions, all_agents_target_actions):
        """
        更新资源智能体的网络
        Args:
            batch: 经验批次
            all_agents_actions: 当前所有智能体的动作
            all_agents_target_actions: 所有智能体目标网络的动作
        """
        (local_states, global_states, global_actions, rewards,
         next_local_states, next_global_states, dones) = batch
        
        # 转换为张量
        local_states = torch.FloatTensor(local_states).to(self.device)
        global_states = torch.FloatTensor(global_states).to(self.device)
        global_actions = torch.FloatTensor(global_actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_local_states = torch.FloatTensor(next_local_states).to(self.device)
        next_global_states = torch.FloatTensor(next_global_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ===== 更新Critic =====
        with torch.no_grad():
            next_global_actions_target = torch.FloatTensor(
                all_agents_target_actions
            ).to(self.device)
            target_q = self.critic_target(next_global_states, next_global_actions_target)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(global_states, global_actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()
        
        # ===== 更新Actor =====
        local_actions = self.actor(local_states)
        global_actions_for_policy = torch.FloatTensor(all_agents_actions).to(self.device)
        
        actor_loss = -self.critic(global_states, global_actions_for_policy).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()
        
        # ===== 软更新目标网络 =====
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        # 记录损失
        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()
    
    def get_target_action(self, state):
        """
        使用目标网络获取动作
        Args:
            state: 状态
        Returns:
            action: 动作
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor_target(state).cpu().numpy().flatten()
        
        return action


if __name__ == "__main__":
    # 测试代码
    print("Testing ResourceAgent...")
    
    state_dim = 12  # 资源层状态维度
    action_dim = 19  # 资源层动作维度
    global_state_dim = 45  # 全局状态维度
    global_action_dim = 52  # 全局动作维度
    
    # 创建资源智能体
    agent = ResourceAgent(
        state_dim, action_dim,
        global_state_dim, global_action_dim
    )
    print(f"Resource Agent created: {agent}")
    
    # 测试获取动作
    state = np.random.randn(state_dim)
    action = agent.get_action(state)
    print(f"Action shape: {action.shape}, range: [{action.min():.3f}, {action.max():.3f}]")
    
    # 测试目标动作
    target_action = agent.get_target_action(state)
    print(f"Target action shape: {target_action.shape}")
    
    print("\nResourceAgent test passed!")

