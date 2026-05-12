"""
sinkhorn_mmot_torch.py — True Multi-Marginal Optimal Transport Sinkhorn solver (GPU)

Implements the IPFP/Sinkhorn algorithm for multi-marginal OT with factorized kernel,
without ever forming the full tensor Γ(i, j_1, ..., j_C).

Key structure:
    - Query marginal: a[P] where P = number of query patches
    - Class marginals: b^c[N_c] for c = 1..C classes
    - Cost matrices: M^c[P, N_c] = 1 - cosine_sim(query_patch_i, support_j)
    
The entropic kernel factorizes:
    K(i, j_1, ..., j_C) = Π_c K^c(i, j_c)  where K^c = exp(-M^c / ε)

The MMOT coupling has the form:
    Γ(i, j_1, ..., j_C) = u_0(i) * Π_c [u_c(j_c) * K^c(i, j_c)]

We only need to track:
    - u_0[P]: query patch scalings
    - u_c[N_c]: class support scalings (one per class)
    - s_c[P] = K^c @ u_c: messages from class c to patches

The pairwise marginal Γ_c(i, j) between patches and class c is:
    Γ_c(i, j) = u_0(i) * p_not_c(i) * K^c(i, j) * u_c(j)
where p_not_c(i) = Π_{d≠c} s_d(i)
"""

import math
from typing import List, Optional, Tuple, Dict, Union

import torch


def sinkhorn_mmot_unbalanced(
    M_list: List[torch.Tensor],
    a: torch.Tensor,
    b_list: List[torch.Tensor],
    reg: float,
    reg_m: float,
    *,
    balanced_query: bool = True,
    numItermax: int = 200,
    stopThr: float = 1e-6,
    verbose: bool = False,
    log: bool = False,
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], dict]]:
    """
    Multi-marginal unbalanced Sinkhorn for one query's patches vs C class supports.
    
    Args:
        M_list: List of C cost matrices, each [P, N_c] in float64
        a: [P] query patch marginal (should sum to 1)
        b_list: List of C class marginals, each [N_c] (should sum to 1)
        reg: Entropic regularization (epsilon)
        reg_m: KL penalty for marginal deviations (lambda)
        balanced_query: If True, enforce query marginal exactly (balanced rows).
                       If False, allow unbalanced query marginal too.
        numItermax: Maximum iterations
        stopThr: Convergence threshold
        verbose: Print convergence info
        log: Return log dict with convergence history
        
    Returns:
        Dict with:
            - 'u0': [P] query scalings
            - 'u_list': List of C tensors, each [N_c] class scalings
            - 's_list': List of C tensors, each [P] class messages
            - 'prod': [P] product of all messages
            - 'Gamma_list': List of C pairwise marginals [P, N_c]
            - 'cost_per_class': [C] transport cost to each class
            - 'mass_per_class': [C] mass sent to each class
    """
    C = len(M_list)
    P = M_list[0].shape[0]
    device = M_list[0].device
    dtype = M_list[0].dtype
    
    # Tau for unbalanced updates
    tau = reg_m / (reg_m + reg) if not math.isinf(reg_m) else 1.0
    
    # Precompute kernels K^c = exp(-M^c / reg)
    K_list = [torch.exp(-M_c / reg) for M_c in M_list]
    
    # Initialize scalings
    u0 = torch.ones(P, device=device, dtype=dtype)
    u_list = [torch.ones(b_c.shape[0], device=device, dtype=dtype) for b_c in b_list]
    
    # For convergence tracking
    err_history = [] if log else None
    tiny = 1e-12
    
    for it in range(numItermax):
        u0_prev = u0.clone()
        u_list_prev = [u_c.clone() for u_c in u_list]
        
        # Step 1: Compute messages s_c = K^c @ u_c for all classes
        s_list = [K_c @ u_c for K_c, u_c in zip(K_list, u_list)]
        
        # Step 2: Compute product prod = s_1 ⊙ s_2 ⊙ ... ⊙ s_C
        prod = torch.ones(P, device=device, dtype=dtype)
        for s_c in s_list:
            prod = prod * s_c
        
        # Step 3: Update query marginal scaling
        if balanced_query:
            # Balanced: u0 = a / (prod + tiny)
            u0 = a / (prod + tiny)
        else:
            # Unbalanced: u0 = (a / (prod + tiny))^tau
            u0 = torch.pow(a / (prod + tiny), tau)
        
        # Step 4: Update each class marginal scaling
        for c in range(C):
            # p_not_c = prod / (s_c + tiny)
            p_not_c = prod / (s_list[c] + tiny)
            
            # t_c = K^c^T @ (u0 ⊙ p_not_c)
            t_c = K_list[c].t() @ (u0 * p_not_c)
            
            # Unbalanced update: u_c = (b_c / (t_c + tiny))^tau
            u_list[c] = torch.pow(b_list[c] / (t_c + tiny), tau)
        
        # Convergence check
        err_u0 = torch.abs(u0 - u0_prev).max() / (torch.abs(u0).max() + tiny)
        err_uc = max(
            torch.abs(u_list[c] - u_list_prev[c]).max() / (torch.abs(u_list[c]).max() + tiny)
            for c in range(C)
        )
        err = max(err_u0.item(), err_uc.item())
        
        if log:
            err_history.append(err)
            if verbose and (it % 10 == 0):
                print(f"MMOT It. {it:4d} | Err {err:.6e}")
        
        if err < stopThr:
            if verbose:
                print(f"MMOT converged at iteration {it}")
            break
    
    # Recompute final messages and product
    s_list = [K_c @ u_c for K_c, u_c in zip(K_list, u_list)]
    prod = torch.ones(P, device=device, dtype=dtype)
    for s_c in s_list:
        prod = prod * s_c
    
    # Compute pairwise marginals Γ_c(i,j) = u0(i) * p_not_c(i) * K^c(i,j) * u_c(j)
    Gamma_list = []
    cost_per_class = torch.zeros(C, device=device, dtype=dtype)
    mass_per_class = torch.zeros(C, device=device, dtype=dtype)
    
    for c in range(C):
        p_not_c = prod / (s_list[c] + tiny)
        # Γ_c = (u0 ⊙ p_not_c)[:,None] ⊙ K^c ⊙ u_c[None,:]
        Gamma_c = (u0 * p_not_c).unsqueeze(1) * K_list[c] * u_list[c].unsqueeze(0)
        Gamma_list.append(Gamma_c)
        
        # Cost and mass
        cost_per_class[c] = (Gamma_c * M_list[c]).sum()
        mass_per_class[c] = Gamma_c.sum()
    
    result = {
        'u0': u0,
        'u_list': u_list,
        's_list': s_list,
        'prod': prod,
        'Gamma_list': Gamma_list,
        'cost_per_class': cost_per_class,
        'mass_per_class': mass_per_class,
    }
    
    if log:
        return result, {'err': err_history}
    return result


def sinkhorn_mmot_unbalanced_batched(
    M_list: List[torch.Tensor],
    a: torch.Tensor,
    b_list: List[torch.Tensor],
    reg: float,
    reg_m: float,
    *,
    balanced_query: bool = True,
    numItermax: int = 100,
    stopThr: float = 1e-6,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Batched multi-marginal unbalanced Sinkhorn for Bq queries vs C class supports.
    
    This processes all query images in parallel (same support set for all).
    
    Args:
        M_list: List of C cost matrices, each [Bq, P, N_c] in float64
        a: [P] query patch marginal (shared across batch, should sum to 1)
        b_list: List of C class marginals, each [N_c] (should sum to 1)
        reg: Entropic regularization (epsilon)
        reg_m: KL penalty for marginal deviations (lambda)
        balanced_query: If True, enforce query marginal exactly (balanced rows).
        numItermax: Maximum iterations
        stopThr: Convergence threshold
        verbose: Print convergence info
        
    Returns:
        Dict with:
            - 'u0': [Bq, P] query scalings
            - 'u_list': List of C tensors, each [Bq, N_c] class scalings
            - 'Gamma_list': List of C pairwise marginals [Bq, P, N_c]
            - 'cost_per_class': [Bq, C] transport cost to each class
            - 'mass_per_class': [Bq, C] mass sent to each class
    """
    C = len(M_list)
    Bq, P, _ = M_list[0].shape
    device = M_list[0].device
    dtype = M_list[0].dtype
    
    # Tau for unbalanced updates
    tau = reg_m / (reg_m + reg) if not math.isinf(reg_m) else 1.0
    
    # Precompute kernels K^c = exp(-M^c / reg)  shape [Bq, P, N_c]
    K_list = [torch.exp(-M_c / reg) for M_c in M_list]
    
    # Expand a to batch: [Bq, P]
    a_batch = a.unsqueeze(0).expand(Bq, P)
    
    # Initialize scalings
    u0 = torch.ones(Bq, P, device=device, dtype=dtype)
    u_list = [torch.ones(Bq, b_c.shape[0], device=device, dtype=dtype) for b_c in b_list]
    
    tiny = 1e-12
    
    for it in range(numItermax):
        u0_prev = u0.clone()
        u_list_prev = [u_c.clone() for u_c in u_list]
        
        # Step 1: Compute messages s_c = K^c @ u_c for all classes
        # K^c: [Bq, P, N_c], u_c: [Bq, N_c] -> s_c: [Bq, P]
        s_list = [torch.bmm(K_c, u_c.unsqueeze(2)).squeeze(2) for K_c, u_c in zip(K_list, u_list)]
        
        # Step 2: Compute product prod = s_1 ⊙ s_2 ⊙ ... ⊙ s_C
        prod = torch.ones(Bq, P, device=device, dtype=dtype)
        for s_c in s_list:
            prod = prod * s_c
        
        # Step 3: Update query marginal scaling
        if balanced_query:
            u0 = a_batch / (prod + tiny)
        else:
            u0 = torch.pow(a_batch / (prod + tiny), tau)
        
        # Step 4: Update each class marginal scaling
        for c in range(C):
            p_not_c = prod / (s_list[c] + tiny)  # [Bq, P]
            
            # t_c = K^c^T @ (u0 ⊙ p_not_c)
            # K^c: [Bq, P, N_c], (u0 * p_not_c): [Bq, P] -> t_c: [Bq, N_c]
            t_c = torch.bmm(K_list[c].transpose(1, 2), (u0 * p_not_c).unsqueeze(2)).squeeze(2)
            
            # Expand b_c to batch for division
            b_c_batch = b_list[c].unsqueeze(0).expand(Bq, -1)
            
            u_list[c] = torch.pow(b_c_batch / (t_c + tiny), tau)
        
        # Convergence check (max across batch)
        err_u0 = (torch.abs(u0 - u0_prev).amax(dim=1) / (torch.abs(u0).amax(dim=1) + tiny)).max()
        err_uc = max(
            (torch.abs(u_list[c] - u_list_prev[c]).amax(dim=1) / (torch.abs(u_list[c]).amax(dim=1) + tiny)).max()
            for c in range(C)
        )
        err = max(err_u0.item(), err_uc.item())
        
        if verbose and (it % 10 == 0):
            print(f"MMOT Batched It. {it:4d} | Err {err:.6e}")
        
        if err < stopThr:
            if verbose:
                print(f"MMOT Batched converged at iteration {it}")
            break
    
    # Recompute final messages and product
    s_list = [torch.bmm(K_c, u_c.unsqueeze(2)).squeeze(2) for K_c, u_c in zip(K_list, u_list)]
    prod = torch.ones(Bq, P, device=device, dtype=dtype)
    for s_c in s_list:
        prod = prod * s_c
    
    # Compute pairwise marginals
    Gamma_list = []
    cost_per_class = torch.zeros(Bq, C, device=device, dtype=dtype)
    mass_per_class = torch.zeros(Bq, C, device=device, dtype=dtype)
    
    for c in range(C):
        p_not_c = prod / (s_list[c] + tiny)  # [Bq, P]
        # Γ_c = (u0 ⊙ p_not_c)[:,:,None] ⊙ K^c ⊙ u_c[:,None,:]
        Gamma_c = (u0 * p_not_c).unsqueeze(2) * K_list[c] * u_list[c].unsqueeze(1)  # [Bq, P, N_c]
        Gamma_list.append(Gamma_c)
        
        cost_per_class[:, c] = (Gamma_c * M_list[c]).sum(dim=(1, 2))
        mass_per_class[:, c] = Gamma_c.sum(dim=(1, 2))
    
    return {
        'u0': u0,
        'u_list': u_list,
        'Gamma_list': Gamma_list,
        'cost_per_class': cost_per_class,
        'mass_per_class': mass_per_class,
    }


def mmot_parts_logits(
    result: Dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute per-class logits from MMOT result.
    
    logit_c = -cost_c / (mass_c + eps)
    
    Args:
        result: Output from sinkhorn_mmot_unbalanced or sinkhorn_mmot_unbalanced_batched
        eps: Small constant for numerical stability
        
    Returns:
        logits: [C] or [Bq, C] depending on input
    """
    cost = result['cost_per_class']
    mass = result['mass_per_class']
    return -cost / (mass + eps)
