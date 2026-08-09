"""Small tensor scorers for dynamically sized candidate collections."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from ._validation import require_floating, require_tensor
from .errors import InvalidTensorValue


def _validate_feature_inputs(
    context: Tensor,
    candidates: Tensor,
    *,
    context_features: int | None = None,
    action_features: int | None = None,
) -> None:
    require_tensor(context, source="context")
    require_tensor(candidates, source="candidates")
    require_floating(context, source="context")
    require_floating(candidates, source="candidates")
    if context.ndim != 1:
        raise InvalidTensorValue(
            "context", f"expected one feature dimension, got {tuple(context.shape)}"
        )
    if candidates.ndim != 2:
        raise InvalidTensorValue(
            "candidates",
            f"expected candidate-feature matrix, got {tuple(candidates.shape)}",
        )
    if context.device != candidates.device:
        raise InvalidTensorValue("inputs", "context and candidates must share a device")
    if context.dtype != candidates.dtype:
        raise InvalidTensorValue("inputs", "context and candidates must share a dtype")
    if context_features is not None and context.shape[0] != context_features:
        raise InvalidTensorValue(
            "context",
            f"expected {context_features} features, got {context.shape[0]}",
        )
    if action_features is not None and candidates.shape[1] != action_features:
        raise InvalidTensorValue(
            "candidates",
            f"expected {action_features} features, got {candidates.shape[1]}",
        )


class DotProductScorer(nn.Module):
    """Score each candidate by its dot product with one context vector."""

    def forward(self, context: Tensor, candidates: Tensor) -> Tensor:
        _validate_feature_inputs(context, candidates)
        if candidates.shape[1] != context.shape[0]:
            raise InvalidTensorValue(
                "candidates",
                f"expected {context.shape[0]} features, got {candidates.shape[1]}",
            )
        return torch.mv(candidates, context)


class MLPScorer(nn.Module):
    """Score concatenated context-candidate features with a two-layer MLP."""

    context_features: int
    action_features: int
    hidden_features: int

    def __init__(
        self,
        context_features: int,
        action_features: int,
        hidden_features: int,
    ) -> None:
        super().__init__()
        for name, value in (
            ("context_features", context_features),
            ("action_features", action_features),
            ("hidden_features", hidden_features),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.context_features = context_features
        self.action_features = action_features
        self.hidden_features = hidden_features
        self.network = nn.Sequential(
            nn.Linear(context_features + action_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, context: Tensor, candidates: Tensor) -> Tensor:
        _validate_feature_inputs(
            context,
            candidates,
            context_features=self.context_features,
            action_features=self.action_features,
        )
        expanded_context = context.unsqueeze(0).expand(candidates.shape[0], -1)
        features = torch.cat((expanded_context, candidates), dim=1)
        return cast(Tensor, self.network(features).squeeze(1))


__all__ = ["DotProductScorer", "MLPScorer"]
