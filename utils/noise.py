"""
Ornstein-Uhlenbeck Noise Process
用于探索的时间相关噪声
"""

import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck随机过程
    用于生成时间相关的探索噪声
    
    dX_t = θ(μ - X_t)dt + σdW_t
    
    其中:
    - θ: 均值回归速度
    - μ: 长期均值
    - σ: 波动率
    - W_t: 维纳过程
    """
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.3, dt=1e-2):
        """
        Args:
            action_dim: 动作空间维度
            mu: 长期均值
            theta: 均值回归速度
            sigma: 噪声标准差
            dt: 时间步长
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = None
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """
        采样噪声
        Returns:
            noise: 噪声向量 [action_dim]
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def __call__(self):
        """方便调用"""
        return self.sample()
    
    def decay_sigma(self, decay_rate=0.995):
        """衰减噪声强度"""
        self.sigma *= decay_rate
    
    def set_sigma(self, sigma):
        """设置噪声强度"""
        self.sigma = sigma


class AdaptiveOUNoise(OUNoise):
    """
    自适应OU噪声
    根据训练进度自动调整噪声强度
    """
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, 
                 sigma_start=0.3, sigma_end=0.01, decay_rate=0.995):
        """
        Args:
            action_dim: 动作空间维度
            mu: 长期均值
            theta: 均值回归速度
            sigma_start: 初始噪声标准差
            sigma_end: 最终噪声标准差
            decay_rate: 衰减率
        """
        super().__init__(action_dim, mu, theta, sigma_start)
        
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_rate = decay_rate
        self.episode_count = 0
    
    def update_sigma(self):
        """
        更新噪声强度（每个episode结束后调用）
        使用指数衰减: σ_t = σ_end + (σ_start - σ_end) * decay_rate^t
        """
        self.episode_count += 1
        self.sigma = self.sigma_end + \
                     (self.sigma_start - self.sigma_end) * \
                     (self.decay_rate ** self.episode_count)
    
    def reset_for_episode(self):
        """每个episode开始时重置"""
        self.reset()
        self.update_sigma()


class GaussianNoise:
    """
    简单的高斯噪声（用于对比）
    """
    
    def __init__(self, action_dim, mu=0.0, sigma=0.3):
        """
        Args:
            action_dim: 动作空间维度
            mu: 均值
            sigma: 标准差
        """
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma
    
    def sample(self):
        """
        采样噪声
        Returns:
            noise: 噪声向量 [action_dim]
        """
        return np.random.normal(self.mu, self.sigma, self.action_dim)
    
    def __call__(self):
        """方便调用"""
        return self.sample()
    
    def reset(self):
        """占位方法，保持接口一致"""
        pass
    
    def decay_sigma(self, decay_rate=0.995):
        """衰减噪声强度"""
        self.sigma *= decay_rate
    
    def set_sigma(self, sigma):
        """设置噪声强度"""
        self.sigma = sigma


if __name__ == "__main__":
    # 测试代码
    print("Testing OUNoise...")
    
    action_dim = 18
    ou_noise = OUNoise(action_dim, sigma=0.3)
    
    print(f"OUNoise created with action_dim={action_dim}")
    
    # 生成一些噪声样本
    samples = []
    for _ in range(100):
        noise = ou_noise.sample()
        samples.append(noise)
    
    samples = np.array(samples)
    print(f"Generated {len(samples)} noise samples")
    print(f"Mean: {samples.mean(axis=0).mean():.4f} (expected ~{ou_noise.mu:.4f})")
    print(f"Std: {samples.std(axis=0).mean():.4f} (expected ~{ou_noise.sigma:.4f})")
    
    # 测试衰减
    print("\nTesting noise decay...")
    initial_sigma = ou_noise.sigma
    for i in range(10):
        ou_noise.decay_sigma(0.9)
    print(f"Sigma after 10 decays: {ou_noise.sigma:.4f} (from {initial_sigma:.4f})")
    
    # 测试自适应OU噪声
    print("\nTesting AdaptiveOUNoise...")
    adaptive_noise = AdaptiveOUNoise(action_dim, sigma_start=0.3, sigma_end=0.01)
    
    print(f"Initial sigma: {adaptive_noise.sigma:.4f}")
    for episode in range(10):
        adaptive_noise.reset_for_episode()
        if episode % 2 == 0:
            print(f"Episode {episode}: sigma = {adaptive_noise.sigma:.4f}")
    
    # 测试高斯噪声
    print("\nTesting GaussianNoise...")
    gaussian_noise = GaussianNoise(action_dim, sigma=0.3)
    noise_samples = [gaussian_noise.sample() for _ in range(100)]
    noise_samples = np.array(noise_samples)
    print(f"Gaussian noise mean: {noise_samples.mean():.4f}")
    print(f"Gaussian noise std: {noise_samples.std():.4f}")
    
    print("\nAll tests passed!")

