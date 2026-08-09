# Adaptive Choice Torch

`adaptive-choice-torch` is the optional PyTorch integration for
`adaptive-choice`. It is a separate distribution so the core runtime remains
dependency-free.

The package provides:

- `TorchChoiceModel`, which adapts tensor encoders and a tensor scorer to the
  scalar `ChoiceModel` protocol;
- `DotProductScorer` and `MLPScorer` building blocks;
- `pad_candidates` and `masked_logits_to_rows` for ordered, dynamically sized
  candidate batches.

```python
from adaptive_choice_torch import DotProductScorer, TorchChoiceModel

model = TorchChoiceModel(
    context_encoder=encode_observation_and_agent,
    action_encoder=encode_actions,
    scorer=DotProductScorer(),
)
```

`TorchChoiceModel.logits` returns ordinary Python floats for the core runtime.
Use `tensor_logits` directly when a downstream training system needs the
original computation graph.

The application owns model mode, tensor placement, input dtypes, and training.
The adapter never calls `eval()` or moves input modules and tensors between
devices. Its scalar boundary intentionally detaches and copies final logits to
the host.
