"""
Experiments package for trust simulation.
"""
from .trust_game import TrustGame, MultiTrustGame
from .dictator_game import DictatorGame, MultiDictatorGame

__all__ = ["TrustGame", "MultiTrustGame", "DictatorGame", "MultiDictatorGame"]