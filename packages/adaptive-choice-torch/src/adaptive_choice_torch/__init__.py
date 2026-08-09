"""Optional PyTorch integration for Adaptive Choice."""

from .batching import PaddedCandidates, masked_logits_to_rows, pad_candidates
from .errors import AdaptiveChoiceTorchError, InvalidTensorValue
from .model import TorchChoiceModel
from .scoring import DotProductScorer, MLPScorer

__version__ = "0.2.0"

__all__ = [
    "AdaptiveChoiceTorchError",
    "DotProductScorer",
    "InvalidTensorValue",
    "MLPScorer",
    "PaddedCandidates",
    "TorchChoiceModel",
    "__version__",
    "masked_logits_to_rows",
    "pad_candidates",
]
