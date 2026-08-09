# PyTorch adapter

Adaptive Choice 0.2 ships PyTorch support as the separate
`adaptive-choice-torch` distribution. The core package does not import PyTorch
and remains usable with only the standard library.

## Install

```bash
python -m pip install adaptive-choice-torch
```

For a source checkout:

```bash
python -m pip install -e . -e ./packages/adaptive-choice-torch
```

## Compose a tensor model

`TorchChoiceModel` needs three application-owned operations:

1. a context encoder mapping `(observation, agent)` to one tensor;
2. an action encoder mapping the ordered candidate sequence to a tensor whose
   first dimension is candidate order;
3. a scorer returning a flat floating tensor with one finite logit per action.

```python
import torch

from adaptive_choice_torch import DotProductScorer, TorchChoiceModel


def encode_context(observation, agent):
    return torch.tensor(
        [observation.visible_quality, agent.price_sensitivity],
        dtype=torch.float32,
    )


def encode_actions(actions):
    return torch.tensor(
        [[action.quality, -action.price] for action in actions],
        dtype=torch.float32,
    )


model = TorchChoiceModel(
    context_encoder=encode_context,
    action_encoder=encode_actions,
    scorer=DotProductScorer(),
)
```

Pass `model` anywhere the core accepts `ChoiceModel`. Its `logits` method uses
`torch.inference_mode()` by default, validates the scorer output, then detaches
and copies the final vector to host-side Python floats. Set
`use_inference_mode=False` only when the forward pass needs normal autograd
behavior.

The adapter never calls `eval()` and never moves modules, context tensors, or
candidate tensors to another device. Model mode, input placement, and input
dtype remain application responsibilities.

## Training boundary

`TorchChoiceModel.tensor_logits` returns the validated tensor without detaching
it or changing gradient mode. A downstream training system may call this method
directly and define its own loss, optimizer, batches, and trajectory storage.
The scalar core deliberately does not retain computation graphs in `Choice` or
`StepResult`.

## MLP scoring

`MLPScorer(context_features, action_features, hidden_features)` concatenates
each action row with the shared context vector and returns one scalar per row.
It is an ordinary `torch.nn.Module`; the application may initialize, move,
serialize, and train it using standard PyTorch mechanisms.

## Dynamic candidate batches

`pad_candidates` converts non-empty tensors with different candidate counts into
a `PaddedCandidates` value:

```python
from adaptive_choice_torch import pad_candidates

batch = pad_candidates((first_candidate_tensor, second_candidate_tensor))
batched_logits = application_batched_scorer(context_batch, batch.values, batch.mask)
scalar_rows = batch.logits_to_rows(batched_logits)
```

`batch.mask` is the authoritative boolean mask. `logits_to_rows` excludes padded
positions and preserves the original row and candidate order. Every row must
select at least one candidate. Non-finite values in real candidate positions are
rejected; padding values are ignored.

These helpers batch tensor computation only. Environments still execute through
the scalar `simulate_step` semantics, so scheduling, RNG streams, outcomes, and
agent updates stay explicit.

`DotProductScorer` and `MLPScorer` implement single-decision candidate scoring.
An application that batches multiple decisions owns its batched scorer and uses
the mask when computing padded logits.

## Validation

Malformed tensor boundaries raise `InvalidTensorValue`, including:

- non-tensor encoder or scorer outputs;
- an action tensor with the wrong candidate count;
- non-flat, non-floating, or non-finite logits;
- scorer inputs with conflicting devices or dtypes;
- masks with the wrong shape or dtype;
- empty rows in a dynamic candidate batch.
