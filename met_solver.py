"""Mutual Evidence Transport solvers.

The solvers here are intentionally small and dependency-light. They operate on
one query/support cost matrix at a time and return the real query-to-support
transport plan used for class scoring and evidence visualization.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _as_vector(x: torch.Tensor, name: str, *, like: torch.Tensor) -> torch.Tensor:
    if x.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    return x.to(device=like.device, dtype=like.dtype)


def _validate_inputs(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cost.dim() != 2:
        raise ValueError("cost must be a 2D tensor")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    cost = cost.clone()
    a = _as_vector(a, "a", like=cost)
    b = _as_vector(b, "b", like=cost)

    if cost.size(0) != a.numel() or cost.size(1) != b.numel():
        raise ValueError("cost shape must match a and b")
    if torch.any(a < 0) or torch.any(b < 0):
        raise ValueError("marginals must be non-negative")
    if float(a.sum()) <= 0 or float(b.sum()) <= 0:
        raise ValueError("marginals must have positive total mass")
    if not torch.isfinite(cost).all():
        raise ValueError("cost contains non-finite values")

    return cost, a, b


def _project_row_caps(plan: torch.Tensor, a: torch.Tensor, tiny: float) -> torch.Tensor:
    row_sum = plan.sum(dim=1)
    scale = torch.minimum(torch.ones_like(row_sum), a / torch.clamp(row_sum, min=tiny))
    return plan * scale.unsqueeze(1)


def _project_col_caps(plan: torch.Tensor, b: torch.Tensor, tiny: float) -> torch.Tensor:
    col_sum = plan.sum(dim=0)
    scale = torch.minimum(torch.ones_like(col_sum), b / torch.clamp(col_sum, min=tiny))
    return plan * scale.unsqueeze(0)


def _project_total_mass(plan: torch.Tensor, target_mass: torch.Tensor, tiny: float) -> torch.Tensor:
    return plan * (target_mass / torch.clamp(plan.sum(), min=tiny))


def met_dykstra_projection(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float = 0.05,
    mass_fraction: float = 0.6,
    num_iter: int = 200,
    tol: float = 1e-7,
    tiny: float = 1e-12,
) -> torch.Tensor:
    """Approximate capacity-constrained MET by KL/Bregman projections.

    The returned plan satisfies, up to numerical tolerance,
    ``plan @ 1 <= a``, ``plan.T @ 1 <= b``, and
    ``plan.sum() ~= mass_fraction * min(a.sum(), b.sum())``.
    """
    cost, a, b = _validate_inputs(cost, a, b, epsilon)
    if not (0.0 < mass_fraction <= 1.0):
        raise ValueError("mass_fraction must be in (0, 1]")
    if num_iter <= 0:
        raise ValueError("num_iter must be positive")

    target_mass = torch.minimum(a.sum(), b.sum()) * float(mass_fraction)
    plan = torch.exp(-cost / float(epsilon)).clamp_min(tiny)
    plan = _project_total_mass(plan, target_mass, tiny)

    for _ in range(num_iter):
        previous = plan
        plan = _project_row_caps(plan, a, tiny)
        plan = _project_col_caps(plan, b, tiny)
        plan = _project_total_mass(plan, target_mass, tiny)

        max_delta = torch.max(torch.abs(plan - previous))
        row_violation = torch.clamp(plan.sum(dim=1) - a, min=0).max()
        col_violation = torch.clamp(plan.sum(dim=0) - b, min=0).max()
        mass_error = torch.abs(plan.sum() - target_mass)
        if max_delta <= tol and row_violation <= tol and col_violation <= tol and mass_error <= tol:
            break

    return plan


def _balanced_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float,
    num_iter: int,
    tiny: float,
) -> torch.Tensor:
    kernel = torch.exp(-cost / float(epsilon)).clamp_min(tiny)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(num_iter):
        u = a / torch.clamp(kernel @ v, min=tiny)
        v = b / torch.clamp(kernel.t() @ u, min=tiny)

    return u.unsqueeze(1) * kernel * v.unsqueeze(0)


def met_dustbin_sinkhorn(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float = 0.05,
    query_reject_cost: float = 0.5,
    support_reject_cost: float = 0.5,
    num_iter: int = 200,
    tiny: float = 1e-12,
) -> torch.Tensor:
    """Dustbin MET fallback using balanced Sinkhorn on an augmented problem."""
    cost, a, b = _validate_inputs(cost, a, b, epsilon)
    if query_reject_cost < 0 or support_reject_cost < 0:
        raise ValueError("reject costs must be non-negative")
    if num_iter <= 0:
        raise ValueError("num_iter must be positive")

    rows, cols = cost.shape
    aug_cost = torch.empty((rows + 1, cols + 1), device=cost.device, dtype=cost.dtype)
    aug_cost[:rows, :cols] = cost
    aug_cost[:rows, cols] = float(query_reject_cost)
    aug_cost[rows, :cols] = float(support_reject_cost)
    aug_cost[rows, cols] = 0.0

    aug_a = torch.cat([a, b.sum().view(1)])
    aug_b = torch.cat([b, a.sum().view(1)])
    plan = _balanced_sinkhorn(
        aug_cost,
        aug_a,
        aug_b,
        epsilon=epsilon,
        num_iter=num_iter,
        tiny=tiny,
    )
    return plan[:rows, :cols]


def met_average_cost(cost: torch.Tensor, plan: torch.Tensor, *, tiny: float = 1e-12) -> torch.Tensor:
    if cost.shape != plan.shape:
        raise ValueError("cost and plan must have the same shape")
    return (cost * plan).sum() / torch.clamp(plan.sum(), min=tiny)
