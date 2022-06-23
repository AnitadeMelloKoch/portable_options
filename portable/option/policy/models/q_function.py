import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead


class SingleSharedBias(nn.Module):
    """
    Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


class LinearQFunction(nn.Module):
    """
    Q function parametrized by soley linear layers
    """
    def __init__(self, in_features, n_actions, hidden_size=64,):
        super().__init__()
        self.q_func = nn.Sequential(
            init_chainer_default(nn.Linear(in_features, hidden_size)),
            init_chainer_default(nn.Linear(hidden_size, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    
    def forward(self, x):
        return self.q_func(x)


def compute_value_loss(
    y: torch.Tensor,
    t: torch.Tensor,
    clip_delta: bool = True,
    batch_accumulator: str = "mean",
) -> torch.Tensor:
    """
    from pfrl
    Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        return F.smooth_l1_loss(y, t, reduction=batch_accumulator)
    else:
        return F.mse_loss(y, t, reduction=batch_accumulator) / 2


def compute_weighted_value_loss(
    y: torch.Tensor,
    t: torch.Tensor,
    weights: torch.Tensor,
    clip_delta: bool = True,
    batch_accumulator: str = "mean",
) -> torch.Tensor:
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        weights (torch.Tensor): Weights for y, t.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        losses = F.smooth_l1_loss(y, t, reduction="none")
    else:
        losses = F.mse_loss(y, t, reduction="none") / 2
    losses = losses.reshape(
        -1,
    )
    weights = weights.to(losses.device)
    loss_sum = torch.sum(losses * weights)
    if batch_accumulator == "mean":
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


def compute_q_learning_loss(exp_batch, y, t, errors_out=None) -> torch.Tensor:
    """Compute the q learning loss for a batch of experiences
    Args:
        exp_batch: batch of experiences
        y: q values
        t: target q values
    Returns:
        Computed loss from the minibatch of experiences
    """
    if errors_out is not None:
        del errors_out[:]
        delta = torch.abs(y - t)
        if delta.ndim == 2:
            delta = torch.sum(delta, dim=1)
        delta = delta.detach().cpu().numpy()
        for e in delta:
            errors_out.append(e)

    if "weights" in exp_batch:
        return compute_weighted_value_loss(
            y,
            t,
            exp_batch["weights"],
            clip_delta=True,
            batch_accumulator="mean",
        )
    else:
        return compute_value_loss(
            y,
            t,
            clip_delta=True,
            batch_accumulator="mean",
        )
