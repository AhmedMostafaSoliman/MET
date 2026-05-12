import math
from typing import Optional, Tuple, Union

import torch


TensorLike = Union[torch.Tensor, float]


def _get_parameter_pair(reg_m: Union[float, Tuple[float, float], torch.Tensor]) -> Tuple[float, float]:
    """
    Mirror POT get_parameter_pair for reg_m. Accepts scalar or pair, returns (reg_m1, reg_m2).
    """
    if isinstance(reg_m, torch.Tensor):
        reg_m = reg_m.detach().cpu().flatten().tolist()
    if isinstance(reg_m, (list, tuple)):
        if len(reg_m) == 0:
            raise ValueError("reg_m must be scalar or pair")
        if len(reg_m) == 1:
            return float(reg_m[0]), float(reg_m[0])
        return float(reg_m[0]), float(reg_m[1])
    else:
        return float(reg_m), float(reg_m)


def sinkhorn_knopp_unbalanced_torch(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    reg_m: Union[float, Tuple[float, float], torch.Tensor],
    *,
    reg_type: str = "kl",
    c: Optional[torch.Tensor] = None,
    warmstart: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    verbose: bool = False,
    log: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Torch implementation of POT's sinkhorn_knopp_unbalanced with support for batched queries.

    Inputs
    - M: [P,N] or [B,P,N] cost matrix
    - a: [P] or [B,P]
    - b: [N] or [B,N]
    - reg: positive float (epsilon)
    - reg_m: float or (float,float) for (reg_m1, reg_m2)

    Behavior matches POT's '_sinkhorn.sinkhorn_knopp_unbalanced' with reg_type='kl' by default.
    When inputs are batched on leading dim B, convergence check uses the max error across the batch.
    """
    if M.dim() == 2:
        return _sinkhorn_knopp_unbalanced_single(
            M, a, b, reg, reg_m, reg_type=reg_type, c=c, warmstart=warmstart,
            numItermax=numItermax, stopThr=stopThr, verbose=verbose, log=log,
        )

    if M.dim() != 3:
        raise ValueError("M must be [P,N] or [B,P,N]")

    B, P, N = M.shape
    device = M.device
    dtype = M.dtype

    # Normalize/reshape a, b to [B,P] and [B,N]
    if a.dim() == 1:
        aB = a.view(1, P).expand(B, P).to(device=device, dtype=dtype)
    elif a.dim() == 2 and a.size(0) == B and a.size(1) == P:
        aB = a.to(device=device, dtype=dtype)
    else:
        raise ValueError("a must be [P] or [B,P]")

    if b.dim() == 1:
        bB = b.view(1, N).expand(B, N).to(device=device, dtype=dtype)
    elif b.dim() == 2 and b.size(0) == B and b.size(1) == N:
        bB = b.to(device=device, dtype=dtype)
    else:
        raise ValueError("b must be [N] or [B,N]")

    reg_m1, reg_m2 = _get_parameter_pair(reg_m)
    fi_1 = 1.0 if math.isinf(reg_m1) else (reg_m1 / (reg_m1 + reg))
    fi_2 = 1.0 if math.isinf(reg_m2) else (reg_m2 / (reg_m2 + reg))

    if reg_type.lower() == "entropy":
        # As in POT: overwrite c by ones
        c_mat = torch.ones((B, P, N), device=device, dtype=dtype)
    else:
        if c is None:
            # reference measure c = a b^T per batch
            c_mat = aB.unsqueeze(2) * bB.unsqueeze(1)  # [B,P,N]
        else:
            if c.dim() == 2:
                c_mat = c.view(1, P, N).expand(B, P, N).to(device=device, dtype=dtype)
            elif c.dim() == 3 and c.size(0) == B:
                c_mat = c.to(device=device, dtype=dtype)
            else:
                raise ValueError("c must be [P,N] or [B,P,N]")

    K = torch.exp(-M / reg) * c_mat

    if warmstart is None:
        u = torch.ones((B, P), device=device, dtype=dtype)
        v = torch.ones((B, N), device=device, dtype=dtype)
    else:
        logu, logv = warmstart
        u = torch.exp(logu).to(device=device, dtype=dtype)
        v = torch.exp(logv).to(device=device, dtype=dtype)

    # Main loop
    one = torch.tensor(1.0, device=device, dtype=dtype)
    err_scalar = float("inf")
    dict_log = {"err": []} if log else None

    for it in range(numItermax):
        uprev = u
        vprev = v

        # u = (a / (K v)) ** fi_1
        Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2)  # [B,P]
        u = torch.pow(aB / Kv, fi_1)

        # v = (b / (K^T u)) ** fi_2
        Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)  # [B,N]
        v = torch.pow(bB / Ktu, fi_2)

        # Numerical issues detection per batch element
        issues = (
            (Ktu == 0.0).any(dim=1)
            | torch.isnan(u).any(dim=1)
            | torch.isnan(v).any(dim=1)
            | torch.isinf(u).any(dim=1)
            | torch.isinf(v).any(dim=1)
        )
        if issues.any():
            # revert offending batches
            mask = issues.unsqueeze(1)
            u = torch.where(mask, uprev, u)
            v = torch.where(mask, vprev, v)
            # and stop
            break

        # Relative errors per batch
        denom_u = torch.maximum(torch.maximum(torch.abs(u).amax(dim=1), torch.abs(uprev).amax(dim=1)), one)
        denom_v = torch.maximum(torch.maximum(torch.abs(v).amax(dim=1), torch.abs(vprev).amax(dim=1)), one)
        err_u = (torch.abs(u - uprev).amax(dim=1)) / denom_u
        err_v = (torch.abs(v - vprev).amax(dim=1)) / denom_v
        err = 0.5 * (err_u + err_v)
        err_scalar = float(err.max().item())

        if log:
            dict_log["err"].append(err_scalar)
            if verbose and (it % 50 == 0):
                # Minimal mimic of POT's prints
                print(f"It. {it:5d} | Err {err_scalar:8e}")

        if err_scalar < stopThr:
            break

    # Transport plan: u[:,None]*K*v[None,:]
    plan = u.unsqueeze(2) * K * v.unsqueeze(1)

    if log:
        log_dict = dict_log
        logu = torch.log(u + 1e-300)
        logv = torch.log(v + 1e-300)
        log_dict.update({"logu": logu, "logv": logv})
        return plan, log_dict
    return plan


def _sinkhorn_knopp_unbalanced_single(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    reg_m: Union[float, Tuple[float, float], torch.Tensor],
    *,
    reg_type: str = "kl",
    c: Optional[torch.Tensor] = None,
    warmstart: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    verbose: bool = False,
    log: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """Non-batched variant fully mirroring POT's update rules."""
    if M.dim() != 2:
        raise ValueError("Single variant expects M [P,N]")

    P, N = M.shape
    device = M.device
    dtype = M.dtype

    reg_m1, reg_m2 = _get_parameter_pair(reg_m)
    fi_1 = 1.0 if math.isinf(reg_m1) else (reg_m1 / (reg_m1 + reg))
    fi_2 = 1.0 if math.isinf(reg_m2) else (reg_m2 / (reg_m2 + reg))

    if reg_type.lower() == "entropy":
        c_mat = torch.ones((P, N), device=device, dtype=dtype)
    else:
        c_mat = a.view(P, 1) * b.view(1, N) if c is None else c.to(device=device, dtype=dtype)

    K = torch.exp(-M / reg) * c_mat

    if warmstart is None:
        u = torch.ones((P,), device=device, dtype=dtype)
        v = torch.ones((N,), device=device, dtype=dtype)
    else:
        logu, logv = warmstart
        u = torch.exp(logu).to(device=device, dtype=dtype)
        v = torch.exp(logv).to(device=device, dtype=dtype)

    dict_log = {"err": []} if log else None
    one = torch.tensor(1.0, device=device, dtype=dtype)

    for it in range(numItermax):
        uprev = u
        vprev = v

        Kv = K @ v
        u = torch.pow(a / Kv, fi_1)

        Ktu = K.t() @ u
        v = torch.pow(b / Ktu, fi_2)

        if (
            (Ktu == 0.0).any()
            or torch.isnan(u).any()
            or torch.isnan(v).any()
            or torch.isinf(u).any()
            or torch.isinf(v).any()
        ):
            if verbose:
                print(f"Numerical errors at iteration {it}")
            u = uprev
            v = vprev
            break

        denom_u = torch.maximum(torch.maximum(torch.abs(u).amax(), torch.abs(uprev).amax()), one)
        denom_v = torch.maximum(torch.maximum(torch.abs(v).amax(), torch.abs(vprev).amax()), one)
        err_u = torch.abs(u - uprev).amax() / denom_u
        err_v = torch.abs(v - vprev).amax() / denom_v
        err = 0.5 * (err_u + err_v)
        if log:
            dict_log["err"].append(float(err.item()))
            if verbose and (it % 50 == 0):
                print(f"It. {it:5d} | Err {float(err.item()):8e}")
        if float(err.item()) < stopThr:
            break

    plan = u.view(P, 1) * K * v.view(1, N)
    if log:
        logu = torch.log(u + 1e-300)
        logv = torch.log(v + 1e-300)
        dict_log.update({"logu": logu, "logv": logv})
        return plan, dict_log
    return plan


def sinkhorn_balanced_torch(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    *,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Classic entropic balanced Sinkhorn in torch.
    Supports batched M: [B,m,n] with a: [m] or [B,m], b: [n] or [B,n].
    """
    if M.dim() == 2:
        return _sinkhorn_balanced_single(M, a, b, reg, numItermax=numItermax, stopThr=stopThr, verbose=verbose)

    if M.dim() != 3:
        raise ValueError("M must be [m,n] or [B,m,n]")

    B, m, n = M.shape
    device = M.device
    dtype = M.dtype

    if a.dim() == 1:
        aB = a.view(1, m).expand(B, m).to(device=device, dtype=dtype)
    else:
        aB = a.to(device=device, dtype=dtype)
    if b.dim() == 1:
        bB = b.view(1, n).expand(B, n).to(device=device, dtype=dtype)
    else:
        bB = b.to(device=device, dtype=dtype)

    K = torch.exp(-M / reg)
    u = torch.ones((B, m), device=device, dtype=dtype)
    v = torch.ones((B, n), device=device, dtype=dtype)

    one = torch.tensor(1.0, device=device, dtype=dtype)

    for it in range(numItermax):
        uprev = u
        vprev = v

        Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
        u = aB / Kv

        Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
        v = bB / Ktu

        denom_u = torch.maximum(torch.maximum(torch.abs(u).amax(dim=1), torch.abs(uprev).amax(dim=1)), one)
        denom_v = torch.maximum(torch.maximum(torch.abs(v).amax(dim=1), torch.abs(vprev).amax(dim=1)), one)
        err_u = torch.abs(u - uprev).amax(dim=1) / denom_u
        err_v = torch.abs(v - vprev).amax(dim=1) / denom_v
        err = 0.5 * (err_u + err_v)
        if float(err.max().item()) < stopThr:
            break

    plan = u.unsqueeze(2) * K * v.unsqueeze(1)
    return plan


def _sinkhorn_balanced_single(
    M: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    reg: float,
    *,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    verbose: bool = False,
) -> torch.Tensor:
    m, n = M.shape
    device = M.device
    dtype = M.dtype

    K = torch.exp(-M / reg)
    u = torch.ones((m,), device=device, dtype=dtype)
    v = torch.ones((n,), device=device, dtype=dtype)

    one = torch.tensor(1.0, device=device, dtype=dtype)
    for it in range(numItermax):
        uprev = u
        vprev = v
        Kv = K @ v
        u = a / Kv
        Ktu = K.t() @ u
        v = b / Ktu

        denom_u = torch.maximum(torch.maximum(torch.abs(u).amax(), torch.abs(uprev).amax()), one)
        denom_v = torch.maximum(torch.maximum(torch.abs(v).amax(), torch.abs(vprev).amax()), one)
        err_u = torch.abs(u - uprev).amax() / denom_u
        err_v = torch.abs(v - vprev).amax() / denom_v
        err = 0.5 * (err_u + err_v)
        if float(err.item()) < stopThr:
            break
    plan = u.view(m, 1) * K * v.view(1, n)
    return plan

