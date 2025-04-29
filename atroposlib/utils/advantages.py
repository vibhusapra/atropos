from typing import Sequence

import torch

from atroposlib.type_definitions import number

TensorLike = torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence]
# Type alias for vector of bools
BoolVector = torch.Tensor


def allclose_to_first(
    values: TensorLike,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    return_vector: bool = False,
) -> BoolVector | bool:
    """
    Check if all tensors in `values` are close to the first tensor `values[0]` using a vectorized approach.

    If `return_vector` is False (default), returns a single boolean indicating whether
    every tensor is close to the first tensor. If `return_vector` is True, returns a list
    of booleans where each element corresponds to whether the respective tensor in
    `values` is close to the first tensor. The first element is always True.

    Args:
        values (torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence]):
            Nested list of values to compare. Must be rectangular, but not necessarily 2D.
        rtol (float, optional): Relative tolerance. Defaults to 1e-05.
        atol (float, optional): Absolute tolerance. Defaults to 1e-08.
        equal_nan (bool, optional): Whether to consider NaNs as equal. Defaults to False.
        return_vector (bool, optional): If True, returns a list of booleans for each comparison.
            Defaults to False.

    Returns:
        bool or BoolVector:
            - If `return_vector` is False, returns True if all tensors are close to the first tensor;
              otherwise, returns False.
            - If `return_vector` is True, returns a 1D tensor of bools where the first element is True
              (as the reference tensor is trivially close to itself), and each subsequent element indicates
              whether the corresponding tensor is close to the first tensor.
    """
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)

    reference = values[0]
    is_close = torch.isclose(
        values, reference, rtol=rtol, atol=atol, equal_nan=equal_nan
    )

    # flatten dimensions after first
    result_vector = torch.all(is_close.view(is_close.size(0), -1), dim=1)

    return result_vector if return_vector else bool(torch.all(result_vector))


def compute_stats(data: Sequence[number | Sequence]) -> dict[str, float]:
    """Compute mean and standard deviation from a possibly jagged nested list.

    This function recursively traverses a nested list of numbers (ints or floats)
    and computes the overall mean and standard deviation of the flattened list.

    Args:
        data (Sequence[number | Sequence]): A possibly jagged nested list of numbers.

    Returns:
        dict: A dictionary with two keys:
            - "mean": The mean of all numerical elements.
            - "var": The variance of all numerical elements.
    """

    def accumulate(x):
        """Recursively accumulate the sum, sum of squares, and count of numerical values.

        Args:
            x (int | float | list): A number or a nested list of numbers.

        Returns:
            tuple: A tuple of three elements:
                - total (float): Sum of all numbers.
                - total_sq (float): Sum of squares of all numbers.
                - count (int): Count of numerical elements.
        """
        if isinstance(x, (int, float)):
            return x, x * x, 1
        elif isinstance(x, list):
            total, total_sq, count = 0.0, 0.0, 0
            for item in x:
                s, ss, c = accumulate(item)
                total += s
                total_sq += ss
                count += c
            return total, total_sq, count
        else:
            raise ValueError(f"Invalid element type encountered: {type(x)}")

    total, total_sq, count = accumulate(data)
    if count == 0:
        raise ValueError("No numerical elements found in the input data.")

    mean = total / count
    variance = total_sq / count - mean * mean
    return {"mean": mean, "var": variance}


def compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns from a 1D vector of rewards.

    Given a list or torch tensor of rewards and a discount factor, this function computes
    the discounted return at each timestep. The discounted return at time t is defined as:
      G_t = rewards[t] + gamma * rewards[t+1] + gamma^2 * rewards[t+2] + ...

    Args:
        rewards (list[float] or torch.Tensor): A 1D list or tensor of rewards.
        gamma (float): The discount factor (should be between 0 and 1).

    Returns:
        list[float]: A list containing the discounted returns for each timestep.
    """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float)
    discounted_returns = torch.empty_like(rewards)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        discounted_returns[t] = running_return

    return discounted_returns


def compute_grpo_process_supervision_advantages(
    rewards: Sequence[Sequence[number]], gamma: float = None, std_tol: float = 1e-8
) -> list[torch.Tensor]:
    """
    Given a (possibly jagged) list of list of rewards, compute advantages for GRPO.

    Args:
        rewards (Sequence[Sequence[number]]): A list of list of rewards. Each inner list
            contains a reward for each "step" in a trajectory, where a "step" is an
            abstract unit of time left up to the modeler to define.
        gamma (float): The discount factor.
        std_tol (float): The tolerance for the standard deviation.

    Returns:
        A list of tensors of advantages.

    Raises:
        ValueError: If the standard deviation of the flattened rewards is smaller than the tolerance.
    """
    stats = compute_stats(rewards)
    mean = stats["mean"]
    std = stats["var"] ** 0.5
    if std < std_tol:
        raise ValueError(f"`std` is smaller than tolerance of {std_tol}.")

    normalized_rewards = [
        (torch.tensor(trajectory) - mean) / std for trajectory in rewards
    ]

    if gamma is None:
        advantages = [
            trajectory.flip(dims=[0]).cumsum(dim=0).flip(dims=[0])
            for trajectory in normalized_rewards
        ]
    else:
        advantages = [
            compute_discounted_returns(trajectory, gamma)
            for trajectory in normalized_rewards
        ]

    return advantages
