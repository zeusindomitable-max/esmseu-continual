
"""
ESMSEU-Continual: Extended Stochastic Metric Space Energy Unification for Continual Learning.
Derived from Juhariah et al. (2025): Unified Heat Kernel and Stochastic Trace Framework.

Bridges stochastic geometry and ML optimization to reduce catastrophic forgetting.
"""

from .regularizer import ESMSEURegularizer
from .continual_trainer import ESMSEUContinualTrainer

__version__ = "0.1.0"
__all__ = ["ESMSEURegularizer", "ESMSEUContinualTrainer"]
