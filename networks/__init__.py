"""Neural Network modules"""
from .actor_network import ActorNetwork, AttentionActorNetwork
from .critic_network import CriticNetwork, DuelingCriticNetwork

__all__ = [
    'ActorNetwork',
    'AttentionActorNetwork',
    'CriticNetwork',
    'DuelingCriticNetwork'
]

