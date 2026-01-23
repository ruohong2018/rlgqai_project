# RLGQAI 使用指南

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd rlgqai_project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 运行所有演示
python demo.py

# 运行特定演示
python demo.py --demo 1  # 快速训练
python demo.py --demo 2  # 参数重要性分析
python demo.py --demo 3  # 智能体动作选择
python demo.py --demo 4  # 环境交互
python demo.py --demo 5  # 模型保存加载
```

### 3. 开始训练

```bash
# 使用默认配置训练
python train.py

# 查看所有参数
python train.py --help
```

## 详细使用说明

### 基础训练

```bash
# 训练500个episode（默认）
python train.py --episodes 500

# 训练1000个episode
python train.py --episodes 1000

# 指定随机种子
python train.py --seed 42

# 使用GPU（如果可用）
python train.py --device cuda
```

### 自定义参数

```bash
# 调整学习率
python train.py --actor-lr 1e-4 --critic-lr 3e-4

# 调整批量大小
python train.py --batch-size 256

# 调整缓冲区大小
python train.py --buffer-size 20000

# 组合多个参数
python train.py --episodes 1000 --batch-size 256 --actor-lr 5e-5
```

### 模型类型和量子比特

```bash
# 训练QVGAN模型
python train.py --model-type QVGAN --num-qubits 4

# 训练QBM模型
python train.py --model-type QBM --num-qubits 6

# 训练QVAE模型
python train.py --model-type QVAE --num-qubits 5
```

### 保存和加载

```bash
# 指定保存目录
python train.py --checkpoint-dir ./my_checkpoints

# 调整保存频率（每50个episode保存一次）
python train.py --save-freq 50

# 从检查点恢复训练
python train.py --load-checkpoint ./checkpoints/episode_500

# 仅评估模式
python train.py --eval-only --load-checkpoint ./checkpoints/final
```

### 日志和监控

```bash
# 指定日志目录
python train.py --log-dir ./my_logs

# 调整日志频率（每5个episode记录一次）
python train.py --log-freq 5
```

## Python API 使用

### 基础使用

```python
from qc_maddpg import QCMADDPG
from config import Config

# 创建算法实例
algorithm = QCMADDPG()

# 训练
algorithm.train(num_episodes=100)

# 保存模型
algorithm.save_models("my_model")

# 加载模型
algorithm.load_models("my_model")
```

### 自定义配置

```python
from config import Config

# 创建自定义配置
cfg = Config()
cfg.NUM_EPISODES = 1000
cfg.BATCH_SIZE = 256
cfg.ACTOR_LR = 5e-5
cfg.CRITIC_LR = 1e-4

# 使用自定义配置
from qc_maddpg import QCMADDPG
algorithm = QCMADDPG(cfg=cfg)
algorithm.train()
```

### 单步交互

```python
from qc_maddpg import QCMADDPG

algorithm = QCMADDPG()

# 重置环境
obs = algorithm.env.reset()

# 选择动作
actions_list, global_action = algorithm.select_actions(obs, add_noise=True)

# 转换为配置
config = algorithm._action_to_config(global_action)

# 执行一步
next_obs, reward, done, info = algorithm.env.step(config)

print(f"Reward: {reward:.4f}")
print(f"Performance: {info['performance']}")
```

### 参数重要性分析

```python
from parameter_analyzer import ParameterAnalyzer

# 创建分析器
param_names = ['circuit_depth', 'learning_rate', 'shot_number']
analyzer = ParameterAnalyzer(param_names)

# 记录性能数据
for config, performance in training_history:
    analyzer.record_performance(config, performance)

# 分析重要性
importance = analyzer.analyze_importance(method='correlation')

# 获取Top-K参数
top_10 = analyzer.get_top_k_params(k=10)

# 打印排名
analyzer.print_importance_ranking(top_k=10)

# 保存结果
analyzer.save_analysis('importance_analysis.json')
```

## 高级功能

### 1. 多智能体独立控制

```python
from agents import QuantumAgent, ClassicalAgent, ResourceAgent

# 创建独立的智能体
quantum_agent = QuantumAgent(
    state_dim=15, 
    action_dim=18,
    global_state_dim=45,
    global_action_dim=52
)

# 获取量子层动作
quantum_state = obs['quantum']
quantum_action = quantum_agent.get_action(quantum_state, noise=None)
```

### 2. 自定义环境

```python
from environments import QuantumAIEnvironment

# 创建自定义环境
env = QuantumAIEnvironment(
    model_type='QVGAN',
    num_qubits=6,
    resource_budget=200000,
    target_fidelity=0.95
)

# 重置和交互
state = env.reset()
next_state, reward, done, info = env.step(config)
```

### 3. 噪声策略调整

```python
from utils import AdaptiveOUNoise

# 创建自适应噪声
noise = AdaptiveOUNoise(
    action_dim=18,
    sigma_start=0.5,
    sigma_end=0.01,
    decay_rate=0.99
)

# 每个episode更新
for episode in range(num_episodes):
    noise.reset_for_episode()  # 重置并更新sigma
    
    # 在episode中使用
    for step in range(max_steps):
        noise_sample = noise.sample()
```

### 4. 经验回放自定义

```python
from utils import PrioritizedReplayBuffer

# 创建优先级回放缓冲区
buffer = PrioritizedReplayBuffer(
    capacity=50000,
    alpha=0.7,
    beta_start=0.5,
    beta_end=1.0
)

# 添加经验
buffer.push(state, action, reward, next_state, done)

# 采样
states, actions, rewards, next_states, dones, indices, weights = buffer.sample(128)

# 更新优先级
td_errors = calculate_td_errors(...)
buffer.update_priorities(indices, td_errors)
```

## 常见问题

### Q1: 训练太慢怎么办？

```bash
# 减少episode数量
python train.py --episodes 100

# 减少每个episode的步数（修改config.py中的MAX_STEPS_PER_EPISODE）

# 减少批量大小
python train.py --batch-size 64

# 使用GPU
python train.py --device cuda
```

### Q2: 内存不足怎么办？

```bash
# 减小缓冲区大小
python train.py --buffer-size 5000

# 减小批量大小
python train.py --batch-size 64
```

### Q3: 如何调整探索强度？

```python
# 在config.py中修改
OU_SIGMA = 0.5  # 增大初始探索
NOISE_DECAY = 0.99  # 减慢衰减
MIN_NOISE = 0.05  # 提高最小噪声
```

### Q4: 如何改变优化目标的权重？

```python
# 在config.py中修改
W_CONVERGENCE = 0.5  # 更重视收敛速度
W_QUALITY = 0.3      # 较少重视质量
W_EFFICIENCY = 0.2   # 较少重视效率
```

### Q5: 如何只调优Top-K重要参数？

```python
from parameter_analyzer import ParameterAnalyzer

# 先进行完整训练获取重要性排名
analyzer = ParameterAnalyzer(all_param_names)
# ... 记录训练数据 ...
importance = analyzer.analyze_importance()
top_20 = analyzer.get_top_k_params(k=20)

# 然后在配置中只包含Top-20参数
# 固定其他参数为默认值
```

## 性能优化建议

### 训练加速

1. 使用GPU（如果可用）
2. 增大批量大小（在内存允许的情况下）
3. 减少状态监控的频率
4. 使用更简单的神经网络架构

### 提高样本效率

1. 增大优先级回放的alpha值
2. 使用更大的经验缓冲区
3. 适当减小探索噪声
4. 增加预热步数

### 提高最终性能

1. 训练更多的episodes
2. 使用更复杂的网络架构
3. 精细调整学习率
4. 使用ensemble方法

## 输出文件说明

### 检查点文件

```
checkpoints/
├── quantum_episode_50.pth    # 量子智能体第50轮检查点
├── classical_episode_50.pth  # 经典智能体第50轮检查点
├── resource_episode_50.pth   # 资源智能体第50轮检查点
├── quantum_final.pth         # 量子智能体最终模型
├── classical_final.pth       # 经典智能体最终模型
└── resource_final.pth        # 资源智能体最终模型
```

### 日志文件

```
logs/
├── training.log              # 训练日志
└── tensorboard/              # TensorBoard日志（如果启用）
```

## 联系和支持

- GitHub Issues: [项目Issues页面]
- Email: support@rlgqai.org
- 文档: [在线文档]

---

最后更新：2024年11月

