"""
RLGQAI Configuration File
配置文件：包含所有超参数和系统配置
"""

class Config:
    """系统配置类"""
    
    # ========== 训练参数 ==========
    NUM_EPISODES = 500
    MAX_STEPS_PER_EPISODE = 50
    BATCH_SIZE = 128
    BUFFER_SIZE = 10000
    WARMUP_STEPS = 1000  # 开始训练前的预热步数
    
    # ========== 网络参数 ==========
    ACTOR_LR = 1e-4
    CRITIC_LR = 3e-4
    GAMMA = 0.99  # 折扣因子
    TAU = 0.01  # 软更新系数
    
    # 网络架构
    ACTOR_HIDDEN_DIMS = [256, 256]
    CRITIC_HIDDEN_DIMS = [512, 512, 256]
    
    # ========== 探索参数 ==========
    OU_SIGMA = 0.3  # OU噪声初始标准差
    OU_THETA = 0.15  # OU噪声均值回归速度
    OU_MU = 0.0  # OU噪声长期均值
    NOISE_DECAY = 0.995  # 噪声衰减率
    MIN_NOISE = 0.01  # 最小噪声水平
    
    # ========== 奖励函数参数 ==========
    W_CONVERGENCE = 0.4  # 收敛速度权重
    W_QUALITY = 0.4  # 生成质量权重
    W_EFFICIENCY = 0.2  # 资源效率权重
    BETA = 2.0  # 指数奖励敏感度参数
    
    # 惩罚系数
    LAMBDA_RESOURCE = 10.0  # 资源超限惩罚系数
    LAMBDA_STABILITY = 1.0  # 参数稳定性惩罚系数
    ACTION_NORM_THRESHOLD = 2.0  # 动作范数阈值
    
    # ========== 优先级经验回放参数 ==========
    PRIORITY_ALPHA = 0.6  # 优先级指数
    PRIORITY_BETA_START = 0.4  # 重要性采样权重初始值
    PRIORITY_BETA_END = 1.0  # 重要性采样权重最终值
    PRIORITY_EPSILON = 1e-6  # 防止零优先级
    
    # ========== 梯度处理参数 ==========
    GRADIENT_CLIP_VALUE = 10.0
    USE_BATCH_NORM = True
    
    # ========== 参数空间定义 ==========
    # 量子层参数 (18维)
    QUANTUM_PARAMS = {
        'circuit_depth': {'min': 1, 'max': 10, 'type': 'int'},
        'entanglement_topology': {'min': 0, 'max': 4, 'type': 'categorical'},  # 0:linear, 1:circular, 2:full, 3:sca, 4:custom
        'num_qubits': {'min': 2, 'max': 10, 'type': 'int'},
        'rotation_x_angle': {'min': 0.0, 'max': 6.28318, 'type': 'float'},
        'rotation_y_angle': {'min': 0.0, 'max': 6.28318, 'type': 'float'},
        'rotation_z_angle': {'min': 0.0, 'max': 6.28318, 'type': 'float'},
        'init_strategy': {'min': 0, 'max': 2, 'type': 'categorical'},  # 0:zero, 1:random, 2:custom
        'measurement_basis': {'min': 0, 'max': 2, 'type': 'categorical'},  # 0:Z, 1:X, 2:Y
        'gate_error_mitigation': {'min': 0, 'max': 1, 'type': 'float'},
        'entanglement_depth': {'min': 1, 'max': 5, 'type': 'int'},
        'ancilla_qubits': {'min': 0, 'max': 3, 'type': 'int'},
        'variational_form': {'min': 0, 'max': 3, 'type': 'categorical'},
        'feature_map': {'min': 0, 'max': 2, 'type': 'categorical'},
        'reps': {'min': 1, 'max': 5, 'type': 'int'},
        'coupling_map_style': {'min': 0, 'max': 2, 'type': 'categorical'},
        'basis_gates_set': {'min': 0, 'max': 2, 'type': 'categorical'},
        'swap_strategy': {'min': 0, 'max': 2, 'type': 'categorical'},
        'approximation_degree': {'min': 0.8, 'max': 1.0, 'type': 'float'},
    }
    
    # 经典层参数 (15维)
    CLASSICAL_PARAMS = {
        'learning_rate': {'min': 1e-5, 'max': 1e-2, 'type': 'float', 'log_scale': True},
        'batch_size': {'min': 16, 'max': 256, 'type': 'int'},
        'optimizer_type': {'min': 0, 'max': 2, 'type': 'categorical'},  # 0:Adam, 1:SGD, 2:RMSprop
        'weight_decay': {'min': 0.0, 'max': 0.01, 'type': 'float'},
        'momentum': {'min': 0.5, 'max': 0.99, 'type': 'float'},
        'gradient_clip_norm': {'min': 0.1, 'max': 10.0, 'type': 'float'},
        'discriminator_lr_ratio': {'min': 0.5, 'max': 2.0, 'type': 'float'},
        'generator_updates': {'min': 1, 'max': 5, 'type': 'int'},
        'discriminator_updates': {'min': 1, 'max': 5, 'type': 'int'},
        'beta1': {'min': 0.5, 'max': 0.999, 'type': 'float'},
        'beta2': {'min': 0.9, 'max': 0.9999, 'type': 'float'},
        'epsilon': {'min': 1e-8, 'max': 1e-6, 'type': 'float', 'log_scale': True},
        'dropout_rate': {'min': 0.0, 'max': 0.5, 'type': 'float'},
        'activation_function': {'min': 0, 'max': 2, 'type': 'categorical'},
        'loss_function': {'min': 0, 'max': 2, 'type': 'categorical'},
    }
    
    # 资源层参数 (19维)
    RESOURCE_PARAMS = {
        'shot_number': {'min': 100, 'max': 10000, 'type': 'int', 'log_scale': True},
        'compilation_level': {'min': 0, 'max': 3, 'type': 'categorical'},
        'error_mitigation_method': {'min': 0, 'max': 3, 'type': 'categorical'},  # 0:none, 1:ZNE, 2:PEC, 3:CDR
        'parallelism_level': {'min': 1, 'max': 8, 'type': 'int'},
        'caching_enabled': {'min': 0, 'max': 1, 'type': 'categorical'},
        'transpiler_seed': {'min': 0, 'max': 1000, 'type': 'int'},
        'optimization_level': {'min': 0, 'max': 3, 'type': 'categorical'},
        'routing_method': {'min': 0, 'max': 2, 'type': 'categorical'},
        'layout_method': {'min': 0, 'max': 2, 'type': 'categorical'},
        'scheduling_method': {'min': 0, 'max': 2, 'type': 'categorical'},
        'memory_allocation': {'min': 512, 'max': 8192, 'type': 'int'},  # MB
        'max_parallel_threads': {'min': 1, 'max': 16, 'type': 'int'},
        'timeout': {'min': 60, 'max': 3600, 'type': 'int'},  # seconds
        'retry_attempts': {'min': 0, 'max': 5, 'type': 'int'},
        'queue_priority': {'min': 0, 'max': 2, 'type': 'categorical'},
        'noise_model_enabled': {'min': 0, 'max': 1, 'type': 'categorical'},
        'measurement_error_mitigation': {'min': 0, 'max': 1, 'type': 'categorical'},
        'readout_error_mitigation': {'min': 0, 'max': 1, 'type': 'categorical'},
        'circuit_cache_size': {'min': 10, 'max': 1000, 'type': 'int'},
    }
    
    # 参数维度
    QUANTUM_DIM = 18
    CLASSICAL_DIM = 15
    RESOURCE_DIM = 19
    TOTAL_PARAM_DIM = QUANTUM_DIM + CLASSICAL_DIM + RESOURCE_DIM  # 52
    
    # ========== 状态空间维度 ==========
    STATE_DIM_QUANTUM = 15
    STATE_DIM_CLASSICAL = 18
    STATE_DIM_RESOURCE = 12
    TOTAL_STATE_DIM = STATE_DIM_QUANTUM + STATE_DIM_CLASSICAL + STATE_DIM_RESOURCE  # 45
    
    # ========== 量子设备配置 ==========
    BACKEND_TYPE = 'simulator'  # 'simulator' or 'real_device'
    SIMULATOR_NAME = 'qasm_simulator'
    REAL_DEVICE_NAME = 'ibmq_quito'
    USE_NOISE_MODEL = True
    
    # ========== 实验配置 ==========
    RANDOM_SEED = 42
    SAVE_FREQUENCY = 50  # 每多少episode保存一次模型
    LOG_FREQUENCY = 10  # 每多少episode记录一次日志
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    
    # ========== 性能目标 ==========
    TARGET_CONVERGENCE_STEPS = 500
    TARGET_FIDELITY = 0.9
    RESOURCE_BUDGET = 100000  # qubit-seconds
    
    @classmethod
    def get_param_bounds(cls):
        """获取所有参数的边界"""
        all_params = {
            **cls.QUANTUM_PARAMS,
            **cls.CLASSICAL_PARAMS,
            **cls.RESOURCE_PARAMS
        }
        
        bounds_min = []
        bounds_max = []
        
        for param_name in sorted(all_params.keys()):
            param = all_params[param_name]
            bounds_min.append(param['min'])
            bounds_max.append(param['max'])
        
        return bounds_min, bounds_max
    
    @classmethod
    def get_param_names(cls):
        """获取所有参数名称（按字母顺序）"""
        all_params = {
            **cls.QUANTUM_PARAMS,
            **cls.CLASSICAL_PARAMS,
            **cls.RESOURCE_PARAMS
        }
        return sorted(all_params.keys())
    
    @classmethod
    def normalize_action(cls, raw_action, param_names=None):
        """将原始动作[-1,1]映射到参数范围"""
        import torch
        import numpy as np
        
        if param_names is None:
            param_names = cls.get_param_names()
        
        all_params = {
            **cls.QUANTUM_PARAMS,
            **cls.CLASSICAL_PARAMS,
            **cls.RESOURCE_PARAMS
        }
        
        if isinstance(raw_action, torch.Tensor):
            normalized = torch.zeros_like(raw_action)
        else:
            normalized = np.zeros_like(raw_action)
        
        for i, param_name in enumerate(param_names):
            if i >= len(raw_action):
                break
            
            param = all_params[param_name]
            min_val = param['min']
            max_val = param['max']
            
            # tanh将输出映射到[-1, 1], 然后映射到[min, max]
            if isinstance(raw_action, torch.Tensor):
                normalized[i] = min_val + (max_val - min_val) * (torch.tanh(raw_action[i]) + 1) / 2
            else:
                normalized[i] = min_val + (max_val - min_val) * (np.tanh(raw_action[i]) + 1) / 2
        
        return normalized


# 默认配置实例
config = Config()

