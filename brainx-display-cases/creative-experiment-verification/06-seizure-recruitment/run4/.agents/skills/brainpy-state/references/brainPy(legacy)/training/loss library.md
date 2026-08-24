# Legacy BrainPy loss library

Use this reference to select a `brainpy.losses` objective or regularizer for a
legacy BrainPy training workflow. Choose from the prediction representation,
target representation, and required reduction before connecting the loss to
`bm.grad`, a legacy trainer, or `bp.optim`.

## Choose a classification representation

The main decision is whether targets are class indices, mutually exclusive
probability vectors, or independent binary labels.

| API | Description |
|---|---|
| `bp.losses.cross_entropy_loss(predicts, targets, weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0)` | Use raw class logits with class-index or supported probability targets; use this full interface for class weights, ignored indices, reduction, or label smoothing. |
| `bp.losses.cross_entropy_sparse(predicts, targets)` | Use logits shaped `(..., n_class)` with sparse integer class indices shaped without the class axis. |
| `bp.losses.cross_entropy_sigmoid(predicts, targets)` | Use logits and same-shaped class-probability targets. |
| `bp.losses.nll_loss(input, target, reduction='mean')` | Use only when `input` already contains log-probabilities and `target` contains class indices. |
| `bp.losses.softmax_cross_entropy(logits, labels)` | Use mutually exclusive classes represented by a same-shaped probability distribution or one-hot labels. |
| `bp.losses.sigmoid_binary_cross_entropy(logits, labels)` | Use independent binary labels for multilabel classification; classes are not mutually exclusive. |
| `bp.losses.binary_logistic_loss(predicts, targets)` | Use a scalar binary score with an integer target of 0 or 1. |
| `bp.losses.multiclass_logistic_loss(label, logits)` | Use one class index with one vector of class logits. Note the legacy argument order: label first. |
| `bp.losses.multi_margin_loss(predicts, targets, margin=1.0, p=1, reduction='mean')` | Use the multiclass hinge objective when the correct class score must exceed other scores by a margin. |

```python
import brainpy as bp
import brainpy.math as bm

logits = bm.array([
    [2.0, -1.0, 0.5],
    [-0.5, 0.2, 1.8],
])
class_ids = bm.array([0, 2])

loss = bp.losses.cross_entropy_loss(logits, class_ids)
assert loss.ndim == 0
```

Pass raw, unnormalized scores to cross-entropy functions. Do not apply softmax
before `cross_entropy_loss`; use `nll_loss` only after producing
log-probabilities. A sparse target removes the final class axis; probability or
multilabel targets retain it.

## Choose a regression loss

Choose the reduction and outlier behavior deliberately; similarly named losses
do not have identical scaling.

| API | Description |
|---|---|
| `bp.losses.l1_loss(logits, targets, reduction='mean')` | Use absolute elementwise error with `'none'`, `'mean'`, or `'sum'` reduction. |
| `bp.losses.mean_absolute_error(x, y, axis=None, reduction='mean')` | Use mean absolute error with an explicit reduction axis when needed. |
| `bp.losses.l2_loss(predicts, targets)` | Use the legacy L2 objective containing the documented `0.5` factor. |
| `bp.losses.mean_squared_error(predicts, targets, axis=None, reduction='mean')` | Use squared error with explicit axes and reduction control. |
| `bp.losses.mean_squared_log_error(predicts, targets, axis=None, reduction='mean')` | Use squared error in the logarithmic value domain. |
| `bp.losses.huber_loss(predicts, targets, delta=1.0)` | Use quadratic behavior near zero and L1-like behavior outside radius `delta`. |
| `bp.losses.log_cosh_loss(predicts, targets)` | Use the smooth log-cosh prediction error. |

Do not interchange `l2_loss` and `mean_squared_error` without checking scale:
one encodes the legacy half-squared L2 convention while the other exposes mean,
sum, or unreduced squared error semantics.

## Use CTC for padded sequence alignment

| API | Description |
|---|---|
| `bp.losses.ctc_loss(logits, logit_paddings, labels, label_paddings, blank_id=0, log_epsilon=-100000.0)` | Use when input and target sequences require Connectionist Temporal Classification; supply padding masks and the blank token explicitly. |
| `bp.losses.ctc_loss_with_forward_probs(logits, logit_paddings, labels, label_paddings, blank_id=0, log_epsilon=-100000.0)` | Use when the CTC objective and its forward probabilities are both required. |

Keep `blank_id` outside the semantic label set and make the padding arrays match
their respective time axes. Use the forward-probability variant only when those
intermediates are consumed or inspected.

## Use callable loss objects

Use the class forms when a trainer or model should retain loss configuration as
a reusable callable.

| API | Description |
|---|---|
| `bp.losses.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` | Store the full cross-entropy configuration for repeated calls. |
| `bp.losses.NLLLoss(reduction='mean')` | Store negative-log-likelihood reduction for repeated calls. |
| `bp.losses.L1Loss(reduction='mean')` | Store L1 reduction for repeated calls. |
| `bp.losses.MAELoss(axis=None, reduction='mean')` | Store mean-absolute-error axis and reduction. |
| `bp.losses.MSELoss(reduction='mean')` | Store mean-squared-error reduction. |

Use either the functional or object form for one objective; do not apply both
to the same predictions.

## Add regularization explicitly

Regularizers receive predictions, errors, labels, or parameter PyTrees rather
than owning the optimizer update.

| API | Description |
|---|---|
| `bp.losses.l2_norm(x, axis=None)` | Compute an L2 norm over an array or supported parameter structure. |
| `bp.losses.mean_absolute(outputs, axis=None)` | Compute mean absolute magnitude for a regularization term. |
| `bp.losses.mean_square(predicts, axis=None)` | Compute mean squared magnitude for a regularization term. |
| `bp.losses.log_cosh(errors)` | Compute log-cosh directly from an error array. |
| `bp.losses.smooth_labels(labels, alpha)` | Apply label smoothing to an existing label representation. |

```python
data_loss = bp.losses.mean_squared_error(predictions, targets)
weight_penalty = 2e-4 * bp.losses.l2_norm(model.train_vars().unique().dict()) ** 2
loss = data_loss + weight_penalty
```

Keep the coefficient visible at the call site. Verify that the selected
parameter collection contains trainable parameters and excludes neuronal State
such as voltage, spikes, and delay buffers.

## Source-backed failures

- Produce a scalar before calling `bm.grad`; use an explicit reduction when a
  loss returns per-sample or per-element values.
- Keep the class axis and target encoding consistent. Do not send one-hot labels
  to a sparse-index API or integer indices to a same-shaped multilabel API.
- Preserve the documented `multiclass_logistic_loss(label, logits)` argument
  order even though most other functions place predictions first.
- Do not use `brainpy.state` metric or Braintools metric APIs in this legacy
  reference; their calling and State contracts differ.

## Official sources

- `https://brainpy.readthedocs.io/apis/losses.html`
- Generated function and loss-class pages linked from that API index.
