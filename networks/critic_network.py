"""
Critic Network Implementation
评论家网络：评估状态-动作对的价值
"""

import torch
import torch.nn as nn
import numpy as np


class CriticNetwork(nn.Module):
    """
    评论家网络 (Critic Network)
    输入：全局状态 + 所有智能体的联合动作
    输出：Q值（状态-动作价值）
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 256], use_batch_norm=True):
        """
        Args:
            state_dim: 全局状态维度
            action_dim: 联合动作维度（所有智能体动作总和）
            hidden_dims: 隐藏层维度列表
            use_batch_norm: 是否使用批归一化
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        # 输入为状态和动作的拼接
        input_dim = state_dim + action_dim
        
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层：输出标量Q值
        self.output_layer = nn.Linear(input_dim, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """
        前向传播
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
        Returns:
            q_value: Q值张量 [batch_size, 1]
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 通过隐藏层
        x = self.hidden_layers(x)
        
        # 输出Q值
        q_value = self.output_layer(x)
        
        return q_value
    
    def get_q_value(self, state, action):
        """
        获取Q值（用于评估）
        Args:
            state: 状态，numpy数组或torch张量
            action: 动作，numpy数组或torch张量
        Returns:
            q_value: 标量Q值
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        with torch.no_grad():
            q_value = self.forward(state, action)
        
        return q_value.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.load_state_dict(torch.load(filepath))


class DuelingCriticNetwork(nn.Module):
    """
    Dueling架构的评论家网络
    将Q值分解为状态价值V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 512, 256], use_batch_norm=True):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            use_batch_norm: 是否使用批归一化
        """
        super(DuelingCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_batch_norm = use_batch_norm
        
        input_dim = state_dim + action_dim
        
        # 共享特征提取层
        shared_layers = []
        for i in range(len(hidden_dims) - 1):
            shared_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_dims[i]))
            shared_layers.append(nn.ReLU())
            input_dim = hidden_dims[i]
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """
        前向传播
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
        Returns:
            q_value: Q值张量 [batch_size, 1]
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 共享特征提取
        features = self.shared_layers(x)
        
        # 计算状态价值和优势函数
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 组合Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_value
    
    def get_q_value(self, state, action):
        """获取Q值"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        with torch.no_grad():
            q_value = self.forward(state, action)
        
        return q_value.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.load_state_dict(torch.load(filepath))


if __name__ == "__main__":
    # 测试代码
    print("Testing CriticNetwork...")
    
    state_dim = 45  # 全局状态维度
    action_dim = 52  # 联合动作维度
    batch_size = 32
    
    # 标准评论家网络
    critic = CriticNetwork(state_dim, action_dim)
    print(f"Critic Network created: state_dim={state_dim}, action_dim={action_dim}")
    
    # 测试前向传播
    dummy_state = torch.randn(batch_size, state_dim)
    dummy_action = torch.randn(batch_size, action_dim)
    q_value = critic(dummy_state, dummy_action)
    print(f"Output Q-value shape: {q_value.shape}")
    print(f"Q-value range: [{q_value.min().item():.3f}, {q_value.max().item():.3f}]")
    
    # 测试get_q_value
    single_state = np.random.randn(state_dim)
    single_action = np.random.randn(action_dim)
    single_q = critic.get_q_value(single_state, single_action)
    print(f"Single Q-value: {single_q:.3f}")
    
    # 测试Dueling评论家网络
    print("\nTesting DuelingCriticNetwork...")
    dueling_critic = DuelingCriticNetwork(state_dim, action_dim)
    q_value = dueling_critic(dummy_state, dummy_action)
    print(f"Dueling Critic output shape: {q_value.shape}")
    
    print("\nAll tests passed!")

