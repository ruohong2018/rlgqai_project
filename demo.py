"""
RLGQAI Demo Script
演示脚本：快速展示系统功能
"""

import numpy as np
from qc_maddpg import QCMADDPG
from parameter_analyzer import ParameterAnalyzer
from config import config


def demo_quick_training():
    """演示快速训练"""
    print("\n" + "="*70)
    print("Demo 1: Quick Training (5 episodes)")
    print("="*70 + "\n")
    
    # 创建算法实例
    algorithm = QCMADDPG()
    
    # 快速训练
    algorithm.train(num_episodes=5)
    
    print("\n✅ Quick training demo completed!")


def demo_parameter_importance():
    """演示参数重要性分析"""
    print("\n" + "="*70)
    print("Demo 2: Parameter Importance Analysis")
    print("="*70 + "\n")
    
    # 创建参数分析器
    param_names = [
        'circuit_depth', 'learning_rate', 'shot_number',
        'batch_size', 'entanglement_topology', 'error_mitigation',
        'optimizer_type', 'compilation_level'
    ]
    
    analyzer = ParameterAnalyzer(param_names)
    
    # 模拟收集性能数据
    print("Simulating parameter tuning process...")
    for i in range(100):
        params = {
            'circuit_depth': np.random.randint(1, 10),
            'learning_rate': 10 ** np.random.uniform(-5, -2),
            'shot_number': np.random.randint(100, 10000),
            'batch_size': np.random.randint(16, 256),
            'entanglement_topology': np.random.randint(0, 4),
            'error_mitigation': np.random.randint(0, 3),
            'optimizer_type': np.random.randint(0, 2),
            'compilation_level': np.random.randint(0, 3)
        }
        
        # 模拟性能（某些参数影响更大）
        performance = (
            0.25 * params['circuit_depth'] +
            0.20 * np.log10(params['shot_number']) +
            0.15 * params['batch_size'] / 100.0 +
            0.10 * (3 - params['error_mitigation']) +
            np.random.randn() * 0.05
        )
        
        analyzer.record_performance(params, performance)
    
    # 分析参数重要性
    print("\nAnalyzing parameter importance...")
    analyzer.analyze_importance(method='correlation')
    analyzer.print_importance_ranking(top_k=8)
    
    print("✅ Parameter importance analysis demo completed!")


def demo_agent_actions():
    """演示智能体动作选择"""
    print("\n" + "="*70)
    print("Demo 3: Multi-Agent Action Selection")
    print("="*70 + "\n")
    
    algorithm = QCMADDPG()
    
    # 获取初始观测
    obs = algorithm.env.reset()
    
    print("State dimensions:")
    print(f"  Quantum:   {obs['quantum'].shape}")
    print(f"  Classical: {obs['classical'].shape}")
    print(f"  Resource:  {obs['resource'].shape}")
    print(f"  Global:    {obs['global'].shape}")
    
    # 选择动作
    print("\nSelecting actions from agents...")
    actions_list, global_action = algorithm.select_actions(obs, add_noise=False)
    
    print(f"\nAction dimensions:")
    print(f"  Quantum Agent:   {actions_list[0].shape} -> {len(actions_list[0])} params")
    print(f"  Classical Agent: {actions_list[1].shape} -> {len(actions_list[1])} params")
    print(f"  Resource Agent:  {actions_list[2].shape} -> {len(actions_list[2])} params")
    print(f"  Global Action:   {global_action.shape} -> {len(global_action)} params total")
    
    # 显示动作范围
    print(f"\nAction value ranges:")
    print(f"  Quantum:   [{actions_list[0].min():.3f}, {actions_list[0].max():.3f}]")
    print(f"  Classical: [{actions_list[1].min():.3f}, {actions_list[1].max():.3f}]")
    print(f"  Resource:  [{actions_list[2].min():.3f}, {actions_list[2].max():.3f}]")
    
    print("\n✅ Agent action selection demo completed!")


def demo_environment_interaction():
    """演示环境交互"""
    print("\n" + "="*70)
    print("Demo 4: Environment Interaction")
    print("="*70 + "\n")
    
    algorithm = QCMADDPG()
    
    # 重置环境
    obs = algorithm.env.reset()
    print("Environment reset")
    print(f"Initial performance: {algorithm.env.get_performance()}")
    
    # 执行几步
    print("\nExecuting 3 steps...")
    for step in range(3):
        # 选择动作
        actions_list, global_action = algorithm.select_actions(obs, add_noise=True)
        
        # 转换为配置
        config = algorithm._action_to_config(global_action)
        
        # 执行
        next_obs, reward, done, info = algorithm.env.step(config)
        
        print(f"\nStep {step + 1}:")
        print(f"  Config: depth={config['circuit_depth']}, "
              f"lr={config['learning_rate']:.2e}, "
              f"shots={config['shot_number']}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Fidelity: {info['performance']['fidelity']:.4f}")
        print(f"  Convergence steps: {info['performance']['convergence_steps']}")
        print(f"  Done: {done}")
        
        if done:
            print("  Episode finished!")
            break
        
        obs = next_obs
    
    print("\n✅ Environment interaction demo completed!")


def demo_model_save_load():
    """演示模型保存和加载"""
    print("\n" + "="*70)
    print("Demo 5: Model Save & Load")
    print("="*70 + "\n")
    
    # 创建并训练一个算法
    print("Creating and training algorithm...")
    algorithm1 = QCMADDPG()
    algorithm1.train(num_episodes=2)
    
    # 保存模型
    print("\nSaving models...")
    algorithm1.save_models("demo_checkpoint")
    
    # 创建新算法并加载
    print("\nCreating new algorithm and loading models...")
    algorithm2 = QCMADDPG()
    algorithm2.load_models("demo_checkpoint")
    
    # 验证加载成功
    print("\nVerifying loaded models...")
    obs = algorithm2.env.reset()
    actions_list, global_action = algorithm2.select_actions(obs, add_noise=False)
    print(f"Successfully generated actions with loaded model")
    print(f"  Global action shape: {global_action.shape}")
    
    print("\n✅ Model save & load demo completed!")


def run_all_demos():
    """运行所有演示"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*20 + "RLGQAI System Demonstration" + " "*21 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    demos = [
        ("Quick Training", demo_quick_training),
        ("Parameter Importance", demo_parameter_importance),
        ("Agent Actions", demo_agent_actions),
        ("Environment Interaction", demo_environment_interaction),
        ("Model Save/Load", demo_model_save_load)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo {i} ({name}) failed with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos):
            input("\nPress Enter to continue to next demo...")
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + " "*18 + "All Demonstrations Completed!" + " "*19 + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RLGQAI Demo Script')
    parser.add_argument('--demo', type=int, choices=[1, 2, 3, 4, 5],
                       help='Run specific demo (1-5), or run all if not specified')
    args = parser.parse_args()
    
    if args.demo:
        demos = {
            1: demo_quick_training,
            2: demo_parameter_importance,
            3: demo_agent_actions,
            4: demo_environment_interaction,
            5: demo_model_save_load
        }
        demos[args.demo]()
    else:
        run_all_demos()

