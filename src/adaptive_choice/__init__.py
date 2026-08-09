"""Adaptive, stochastic, preference-conditioned choice for simulations.

The package owns the decision boundary while applications retain their own
world, action, outcome, and agent domain types.
"""

from .errors import (
    ActionCountMismatch,
    AdaptiveChoiceError,
    InvalidChoiceIndex,
    InvalidLogits,
    InvalidProbabilityDistribution,
    InvalidRandomValue,
    NoLegalActions,
)
from .protocols import (
    AgentUpdater,
    ChoiceModel,
    Environment,
    Observer,
    RandomGenerator,
    Sampler,
)
from .runner import DecisionSystem, simulate_step
from .sampling import ArgmaxSampler, SoftmaxSampler
from .types import Choice, StepResult

__version__ = "0.2.0"

__all__ = [
    "ActionCountMismatch",
    "AdaptiveChoiceError",
    "AgentUpdater",
    "ArgmaxSampler",
    "Choice",
    "ChoiceModel",
    "DecisionSystem",
    "Environment",
    "InvalidChoiceIndex",
    "InvalidLogits",
    "InvalidProbabilityDistribution",
    "InvalidRandomValue",
    "NoLegalActions",
    "Observer",
    "RandomGenerator",
    "Sampler",
    "SoftmaxSampler",
    "StepResult",
    "__version__",
    "simulate_step",
]
