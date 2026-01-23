"""
Parameter Importance Analyzer
参数重要性分析模块
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class ParameterAnalyzer:
    """
    参数重要性分析器
    使用以下方法分析参数重要性：
    1. 量子Fisher信息矩阵（量子参数）
    2. Shapley值方法（经典和资源参数）
    """
    
    def __init__(self, param_names: List[str]):
        """
        Args:
            param_names: 参数名称列表
        """
        self.param_names = param_names
        self.importance_scores = {}
        self.performance_history = []
    
    def record_performance(self, params: Dict, performance: float):
        """
        记录参数配置和对应的性能
        Args:
            params: 参数配置
            performance: 性能指标
        """
        self.performance_history.append({
            'params': params.copy(),
            'performance': performance
        })
    
    def analyze_importance(self, method='correlation') -> Dict[str, float]:
        """
        分析参数重要性
        Args:
            method: 分析方法 ('correlation', 'shapley', 'qfim')
        Returns:
            importance_scores: 参数重要性分数字典
        """
        if not self.performance_history:
            raise ValueError("No performance history recorded!")
        
        if method == 'correlation':
            return self._correlation_analysis()
        elif method == 'shapley':
            return self._shapley_analysis()
        elif method == 'qfim':
            return self._qfim_analysis()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _correlation_analysis(self) -> Dict[str, float]:
        """
        基于相关性的参数重要性分析
        计算每个参数与性能指标的相关系数
        """
        param_values = defaultdict(list)
        performances = []
        
        for record in self.performance_history:
            for param_name, param_value in record['params'].items():
                param_values[param_name].append(param_value)
            performances.append(record['performance'])
        
        performances = np.array(performances)
        importance_scores = {}
        
        for param_name in self.param_names:
            if param_name in param_values:
                values = np.array(param_values[param_name])
                
                # 计算Pearson相关系数
                if len(values) > 1 and np.std(values) > 0 and np.std(performances) > 0:
                    correlation = np.corrcoef(values, performances)[0, 1]
                    importance_scores[param_name] = abs(correlation)
                else:
                    importance_scores[param_name] = 0.0
            else:
                importance_scores[param_name] = 0.0
        
        # 归一化
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def _shapley_analysis(self) -> Dict[str, float]:
        """
        基于Shapley值的参数重要性分析
        评估每个参数的边际贡献
        
        简化版本：使用采样方法估计Shapley值
        """
        n_samples = min(len(self.performance_history), 100)
        importance_scores = {name: 0.0 for name in self.param_names}
        
        # 采样评估边际贡献
        for _ in range(n_samples):
            # 随机选择两个配置
            if len(self.performance_history) < 2:
                break
            
            idx1, idx2 = np.random.choice(len(self.performance_history), 2, replace=False)
            config1 = self.performance_history[idx1]
            config2 = self.performance_history[idx2]
            
            perf_diff = abs(config1['performance'] - config2['performance'])
            
            # 计算参数差异
            for param_name in self.param_names:
                if param_name in config1['params'] and param_name in config2['params']:
                    param_diff = abs(config1['params'][param_name] - 
                                   config2['params'][param_name])
                    
                    # 边际贡献估计
                    if param_diff > 0:
                        contribution = perf_diff * param_diff
                        importance_scores[param_name] += contribution
        
        # 归一化
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def _qfim_analysis(self) -> Dict[str, float]:
        """
        基于量子Fisher信息矩阵的参数重要性分析
        
        简化版本：使用参数敏感度估计
        """
        importance_scores = {}
        
        # 对每个参数计算敏感度
        for param_name in self.param_names:
            sensitivities = []
            
            # 查找相邻配置对
            for i in range(len(self.performance_history) - 1):
                config1 = self.performance_history[i]
                config2 = self.performance_history[i + 1]
                
                if param_name in config1['params'] and param_name in config2['params']:
                    param_change = abs(config2['params'][param_name] - 
                                     config1['params'][param_name])
                    perf_change = abs(config2['performance'] - 
                                    config1['performance'])
                    
                    if param_change > 1e-6:
                        sensitivity = perf_change / param_change
                        sensitivities.append(sensitivity)
            
            # 平均敏感度作为重要性分数
            if sensitivities:
                importance_scores[param_name] = np.mean(sensitivities)
            else:
                importance_scores[param_name] = 0.0
        
        # 归一化
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def get_top_k_params(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        获取Top-K重要参数
        Args:
            k: 返回的参数数量
        Returns:
            top_params: (参数名, 重要性分数)列表
        """
        if not self.importance_scores:
            raise ValueError("Must run analyze_importance() first!")
        
        sorted_params = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_params[:k]
    
    def print_importance_ranking(self, top_k: int = 10):
        """
        打印参数重要性排名
        Args:
            top_k: 显示前多少个参数
        """
        top_params = self.get_top_k_params(top_k)
        
        print(f"\n{'='*60}")
        print(f"Top-{top_k} Parameter Importance Ranking")
        print(f"{'='*60}")
        print(f"{'Rank':<6}{'Parameter Name':<30}{'Importance Score':<20}")
        print("-" * 60)
        
        for rank, (param_name, score) in enumerate(top_params, 1):
            print(f"{rank:<6}{param_name:<30}{score:<.6f}")
        
        print(f"{'='*60}\n")
    
    def save_analysis(self, filepath: str):
        """
        保存分析结果
        Args:
            filepath: 保存路径
        """
        import json
        
        data = {
            'param_names': self.param_names,
            'importance_scores': self.importance_scores,
            'num_records': len(self.performance_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Analysis saved to: {filepath}")
    
    def load_analysis(self, filepath: str):
        """
        加载分析结果
        Args:
            filepath: 加载路径
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.param_names = data['param_names']
        self.importance_scores = data['importance_scores']
        
        print(f"Analysis loaded from: {filepath}")


if __name__ == "__main__":
    # 测试参数分析器
    print("Testing ParameterAnalyzer...")
    
    # 创建分析器
    param_names = ['circuit_depth', 'learning_rate', 'shot_number', 
                   'batch_size', 'entanglement_topology']
    analyzer = ParameterAnalyzer(param_names)
    
    # 模拟记录一些性能数据
    print("\nSimulating performance records...")
    for i in range(50):
        params = {
            'circuit_depth': np.random.randint(1, 10),
            'learning_rate': np.random.uniform(1e-5, 1e-2),
            'shot_number': np.random.randint(100, 10000),
            'batch_size': np.random.randint(16, 256),
            'entanglement_topology': np.random.randint(0, 4)
        }
        
        # 模拟性能（与某些参数相关）
        performance = (0.3 * params['circuit_depth'] + 
                      0.4 * np.log10(params['shot_number']) +
                      0.2 * params['batch_size'] / 100.0 +
                      np.random.randn() * 0.1)
        
        analyzer.record_performance(params, performance)
    
    # 相关性分析
    print("\nRunning correlation analysis...")
    importance = analyzer.analyze_importance(method='correlation')
    analyzer.print_importance_ranking(top_k=5)
    
    # Shapley值分析
    print("\nRunning Shapley analysis...")
    importance = analyzer.analyze_importance(method='shapley')
    analyzer.print_importance_ranking(top_k=5)
    
    # 获取Top-3参数
    top_3 = analyzer.get_top_k_params(k=3)
    print("Top-3 most important parameters:")
    for rank, (name, score) in enumerate(top_3, 1):
        print(f"  {rank}. {name}: {score:.4f}")
    
    print("\nParameterAnalyzer test passed!")

