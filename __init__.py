"""
RLGQAI: Reinforcement Learning for Generative Quantum AI Auto-Tuning
利用强化学习为生成式量子AI系统自动调优
"""

__version__ = "1.0.0"
__author__ = "RLGQAI Team"
__description__ = "Auto-tuning framework for generative quantum AI systems using multi-agent reinforcement learning"

from .qc_maddpg import QCMADDPG
from .parameter_analyzer import ParameterAnalyzer

__all__ = [
    'QCMADDPG',
    'ParameterAnalyzer',
]

