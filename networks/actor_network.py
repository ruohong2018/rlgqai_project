"""
Actor Network Implementation
演员网络：输出确定性的参数配置动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorNetwork(nn.Module):
    """
    演员网络 (Actor Network)
    输入：智能体的局部状态观测
    输出：连续动作值（参数配置）
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], use_batch_norm=True):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            use_batch_norm: 是否使用批归一化
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_batch_norm = use_batch_norm
        
        # 构建网络层
        layers = []
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(input_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        前向传播
        Args:
            state: 状态张量 [batch_size, state_dim]
        Returns:
            action: 动作张量 [batch_size, action_dim], 范围约为[-1, 1]
        """
        x = self.hidden_layers(state)
        action = torch.tanh(self.output_layer(x))  # tanh输出[-1, 1]
        return action
    
    def get_action(self, state):
        """
        获取动作（用于执行阶段）
        Args:
            state: 状态，numpy数组或torch张量
        Returns:
            action: numpy数组
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.forward(state)
        
        return action.cpu().numpy().flatten()
    
    def save(self, filepath):
        """保存模型"""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.load_state_dict(torch.load(filepath))


class AttentionActorNetwork(ActorNetwork):
    """
    带注意力机制的演员网络
    用于处理量子比特状态聚合
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 use_batch_norm=True, attention_dim=64):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度
            use_batch_norm: 是否使用批归一化
            attention_dim: 注意力机制隐藏维度
        """
        super(AttentionActorNetwork, self).__init__(
            state_dim, action_dim, hidden_dims, use_batch_norm
        )
        
        self.attention_dim = attention_dim
        
        # 注意力层
        self.attention_key = nn.Linear(state_dim, attention_dim)
        self.attention_query = nn.Linear(state_dim, attention_dim)
        self.attention_value = nn.Linear(state_dim, attention_dim)
        
        # 重新定义输入层以适应注意力输出
        self.attention_output = nn.Linear(attention_dim, state_dim)
    
    def apply_attention(self, state):
        """
        应用注意力机制
        Args:
            state: [batch_size, state_dim]
        Returns:
            attended_state: [batch_size, state_dim]
        """
        # 计算注意力分数
        key = self.attention_key(state)
        query = self.attention_query(state)
        value = self.attention_value(state)
        
        # 缩放点积注意力
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, value)
        attended_state = self.attention_output(attended)
        
        return attended_state + state  # 残差连接
    
    def forward(self, state):
        """
        前向传播（带注意力）
        Args:
            state: 状态张量 [batch_size, state_dim]
        Returns:
            action: 动作张量 [batch_size, action_dim]
        """
        # 应用注意力机制
        attended_state = self.apply_attention(state)
        
        # 通过隐藏层
        x = self.hidden_layers(attended_state)
        
        # 输出层
        action = torch.tanh(self.output_layer(x))
        
        return action


if __name__ == "__main__":
    # 测试代码
    print("Testing ActorNetwork...")
    
    state_dim = 15
    action_dim = 18
    batch_size = 32
    
    # 标准演员网络
    actor = ActorNetwork(state_dim, action_dim)
    print(f"Actor Network created: state_dim={state_dim}, action_dim={action_dim}")
    
    # 测试前向传播
    dummy_state = torch.randn(batch_size, state_dim)
    action = actor(dummy_state)
    print(f"Output action shape: {action.shape}")
    print(f"Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    
    # 测试get_action
    single_state = np.random.randn(state_dim)
    single_action = actor.get_action(single_state)
    print(f"Single action shape: {single_action.shape}")
    
    # 测试注意力演员网络
    print("\nTesting AttentionActorNetwork...")
    attention_actor = AttentionActorNetwork(state_dim, action_dim)
    action = attention_actor(dummy_state)
    print(f"Attention Actor output shape: {action.shape}")
    
    print("\nAll tests passed!")

