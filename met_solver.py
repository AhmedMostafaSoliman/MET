"""Mutual Evidence Transport solvers.

The solvers here are intentionally small and dependency-light. They operate on
one query/support cost matrix at a time and return the real query-to-support
transport plan used for class scoring and evidence visualization.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

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


def _validate_met_inputs(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    mass_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cost, a, b = _validate_inputs(cost, a, b, epsilon)

    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("a or b contains non-finite values")
    if not (0.0 < mass_fraction <= 1.0):
        raise ValueError("mass_fraction must be in (0, 1]")

    return cost, a, b


def _log_project_total_mass(log_z: torch.Tensor, target_mass: torch.Tensor) -> torch.Tensor:
    """KL projection onto {gamma: gamma.sum() = target_mass} in log-space."""
    log_total = torch.logsumexp(log_z.reshape(-1), dim=0)
    return log_z + torch.log(target_mass) - log_total


def _log_project_row_caps(log_z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """KL projection onto {gamma: gamma @ 1 <= a} in log-space."""
    row_log_mass = torch.logsumexp(log_z, dim=1)
    log_scale = torch.minimum(torch.zeros_like(row_log_mass), torch.log(a) - row_log_mass)
    return log_z + log_scale[:, None]


def _log_project_col_caps(log_z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """KL projection onto {gamma: gamma.T @ 1 <= b} in log-space."""
    col_log_mass = torch.logsumexp(log_z, dim=0)
    log_scale = torch.minimum(torch.zeros_like(col_log_mass), torch.log(b) - col_log_mass)
    return log_z + log_scale[None, :]


def _met_dykstra_positive_caps(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float,
    mass_fraction: float,
    num_iter: int,
    tol: float,
    check_every: int,
    return_info: bool,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
    """Run log-space KL-Dykstra when all capacities are strictly positive."""
    target_mass = torch.minimum(a.sum(), b.sum()) * float(mass_fraction)
    if float(target_mass) <= 0:
        raise ValueError("target_mass must be positive")

    log_k = -cost / float(epsilon)
    log_k = log_k - torch.max(log_k)
    log_plan = _log_project_total_mass(log_k, target_mass)

    log_q_row = torch.zeros_like(log_plan)
    log_q_col = torch.zeros_like(log_plan)
    log_q_mass = torch.zeros_like(log_plan)

    converged = False
    last_max_delta = None
    last_row_violation = None
    last_col_violation = None
    last_mass_error = None

    for it in range(1, num_iter + 1):
        should_check = (it == 1) or (it % check_every == 0) or (it == num_iter)
        if should_check:
            previous_plan = torch.exp(log_plan)

        log_y = log_plan + log_q_row
        log_new = _log_project_row_caps(log_y, a)
        log_q_row = log_y - log_new
        log_plan = log_new

        log_y = log_plan + log_q_col
        log_new = _log_project_col_caps(log_y, b)
        log_q_col = log_y - log_new
        log_plan = log_new

        log_y = log_plan + log_q_mass
        log_new = _log_project_total_mass(log_y, target_mass)
        log_q_mass = log_y - log_new
        log_plan = log_new

        if should_check:
            plan = torch.exp(log_plan)
            row_violation = torch.clamp(plan.sum(dim=1) - a, min=0).max()
            col_violation = torch.clamp(plan.sum(dim=0) - b, min=0).max()
            mass_error = torch.abs(plan.sum() - target_mass)
            max_delta = torch.max(torch.abs(plan - previous_plan))

            last_max_delta = max_delta
            last_row_violation = row_violation
            last_col_violation = col_violation
            last_mass_error = mass_error

            if max_delta <= tol and row_violation <= tol and col_violation <= tol and mass_error <= tol:
                converged = True
                break

    plan = torch.exp(log_plan)

    if return_info:
        info = {
            "converged": converged,
            "num_iter": it,
            "target_mass": target_mass.detach(),
            "plan_mass": plan.sum().detach(),
            "max_delta": None if last_max_delta is None else last_max_delta.detach(),
            "row_violation": None if last_row_violation is None else last_row_violation.detach(),
            "col_violation": None if last_col_violation is None else last_col_violation.detach(),
            "mass_error": None if last_mass_error is None else last_mass_error.detach(),
        }
        return plan, info

    return plan


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


def met_dykstra_projection_corrected(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    epsilon: float = 0.05,
    mass_fraction: float = 0.6,
    num_iter: int = 200,
    tol: float = 1e-7,
    check_every: int = 5,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, object]]]:
    """Corrected KL-Dykstra MET solver with explicit log-space corrections.

    Solves the entropic fixed-mass common-submeasure problem subject to row
    caps, column caps, and fixed total evidence mass. This is intentionally
    separate from ``met_dykstra_projection`` so existing experiments keep their
    previous solver behavior.
    """
    if num_iter <= 0:
        raise ValueError("num_iter must be positive")
    if check_every <= 0:
        raise ValueError("check_every must be positive")

    cost, a, b = _validate_met_inputs(cost, a, b, epsilon, mass_fraction)

    active_rows = a > 0
    active_cols = b > 0
    if not active_rows.any():
        raise ValueError("all query capacities are zero")
    if not active_cols.any():
        raise ValueError("all support capacities are zero")

    if active_rows.all() and active_cols.all():
        return _met_dykstra_positive_caps(
            cost,
            a,
            b,
            epsilon=epsilon,
            mass_fraction=mass_fraction,
            num_iter=num_iter,
            tol=tol,
            check_every=check_every,
            return_info=return_info,
        )

    rows = active_rows.nonzero(as_tuple=True)[0]
    cols = active_cols.nonzero(as_tuple=True)[0]
    sub_cost = cost[rows[:, None], cols[None, :]]
    sub_a = a[rows]
    sub_b = b[cols]

    result = _met_dykstra_positive_caps(
        sub_cost,
        sub_a,
        sub_b,
        epsilon=epsilon,
        mass_fraction=mass_fraction,
        num_iter=num_iter,
        tol=tol,
        check_every=check_every,
        return_info=return_info,
    )

    if return_info:
        sub_plan, info = result
    else:
        sub_plan = result
        info = None

    plan = torch.zeros_like(cost)
    plan[rows[:, None], cols[None, :]] = sub_plan

    if return_info:
        return plan, info
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
