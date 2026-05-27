"""Test-Time Augmentation (TTA) inference for OTNet — no FT, no pseudo-labels.

For each query, average MET classification logits across K augmented query views.
The support side is unchanged (single MET solve per view-pair).

Why this might work where everything else didn't:
    - TTA is a well-known cheap inference trick that often delivers +0.5–1pp.
    - It does NOT modify the backbone or the support distribution; only the
      query features are perturbed and the logits averaged.
    - Failure modes of the other methods I tried (LOO overfit, IMSR pseudo-label
      poisoning, MAP signal too small) do not apply here.

Paired baseline-vs-TTA evaluation, same support/query batch per episode.

Knobs:
    --tta_k                    number of augmented views (incl. clean view)
    --tta_aug_strength         strength of the crop+flip+noise augmentation
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


TRUE_VALUES = {"1","true","yes","y","on"}
FALSE_VALUES = {"0","false","no","n","off"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES: return True
    if text in FALSE_VALUES: return False
    raise ValueError(f"Cannot parse boolean: {value!r}")


def append_jsonl_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        h.write(json.dumps(payload, sort_keys=True) + "\n")
        h.flush(); os.fsync(h.fileno())
        fcntl.flock(h, fcntl.LOCK_UN)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def set_all_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EvalSummary:
    baseline_accuracy: float
    candidate_accuracy: float
    delta_accuracy: float
    delta_accuracy_pp: float
    baseline_correct: int
    candidate_correct: int
    total_queries: int
    episodes: int
    candidate_wins: int
    candidate_ties: int
    candidate_losses: int


def tensor_augment(images: torch.Tensor, strength: float) -> torch.Tensor:
    """Random crop + flip + light gaussian noise. Same shape as input."""
    if strength <= 0.0:
        return images
    batch, channels, h, w = images.shape
    min_scale = max(0.65, 1.0 - 0.3 * min(strength, 1.0))
    out = []
    for img in images:
        scale = float(torch.empty((), device=images.device).uniform_(min_scale, 1.0).item())
        ch = max(1, min(h, int(round(h*scale))))
        cw = max(1, min(w, int(round(w*scale))))
        t_lim = max(0, h-ch); l_lim = max(0, w-cw)
        t = int(torch.randint(0, t_lim+1, (), device=images.device).item()) if t_lim else 0
        l = int(torch.randint(0, l_lim+1, (), device=images.device).item()) if l_lim else 0
        crop = img[:, t:t+ch, l:l+cw].unsqueeze(0)
        crop = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        if bool(torch.rand((), device=images.device).item() < 0.5):
            crop = torch.flip(crop, dims=[2])
        out.append(crop)
    out_t = torch.stack(out, dim=0)
    noise_std = 0.01 * min(strength, 1.0)
    if noise_std > 0:
        out_t = out_t + noise_std * torch.randn_like(out_t)
    return out_t


def tta_forward(
    model: torch.nn.Module,
    support_imgs: torch.Tensor,
    support_labels: torch.Tensor,
    query_imgs: torch.Tensor,
    *,
    K: int,
    aug_strength: float,
    avg_mode: str,
    device: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Average logits/probs across K query views (incl. clean view).

    Support is processed ONCE; only queries are augmented per view.

    avg_mode:
      "logits": average raw logits.
      "probs":  softmax each set of logits, average probabilities, then take log.
                Often more robust because softmax bounds.
    """
    K = max(1, int(K))
    model.process_support_set(support_imgs.to(device), support_labels.to(device))
    all_logits: List[torch.Tensor] = []
    # First view is CLEAN
    all_logits.append(model(query_imgs.to(device)).detach())
    for k in range(1, K):
        view = tensor_augment(query_imgs, float(aug_strength))
        all_logits.append(model(view.to(device)).detach())
    stack = torch.stack(all_logits, dim=0)  # [K, Bq, n_way]
    if avg_mode == "probs":
        probs = F.softmax(stack, dim=2).mean(dim=0)
        result = torch.log(probs.clamp_min(1e-12))
    else:
        result = stack.mean(dim=0)
    return result, {
        "K": K, "avg_mode": avg_mode,
        "agreement_rate": float((stack.argmax(dim=2)[0] == stack.argmax(dim=2)).float().mean()),
    }


def load_runtime_modules() -> Any:
    import uot_superpixels as uot
    return uot


def build_dataset(uot: Any, dataset: str, transform: Any, args: argparse.Namespace) -> Any:
    if dataset == "CUB": return uot.CUB(split="test", training=False, transform=transform)
    if dataset == "Plant-Disease": return uot.FastWrapFewShotDataset(uot.MyDataSet("/home_old/ahmedm04/few_shot_ds/Plant-Disease/Plant-Disease", transform=transform))
    if dataset == "BCCD_WBC": return uot.FastWrapFewShotDataset(uot.MyDataSet("/home_old/ahmedm04/few_shot_ds/BCCD_WBC/BCCD_WBC", transform=transform))
    if dataset == "ChestX": return uot.FastWrapFewShotDataset(uot.ChestX("/home_old/ahmedm04/few_shot_ds/ChestX", transform=transform))
    if dataset == "ISIC": return uot.FastWrapFewShotDataset(uot.ISICDataset("/home_old/ahmedm04/few_shot_ds/ISIC2018", transform=transform))
    if dataset == "EUROSAT": return uot.FastWrapFewShotDataset(uot.MyDataSet("/home_old/ahmedm04/few_shot_ds/EUROSAT/EUROSAT", transform=transform))
    if dataset == "HEp": return uot.FastWrapFewShotDataset(uot.MyDataSet("/home_old/ahmedm04/few_shot_ds/HEp-Dataset/HEp-Dataset", transform=transform))
    raise ValueError(f"Unknown dataset for tta_inference: {dataset}")


def model_kwargs(args, patch_size, pixel_mean, pixel_std):
    return {
        "patch_size": patch_size,
        "reg_eps": args.reg_eps, "reg_mass": args.reg_mass,
        "alpha_global": args.alpha_global,
        "transport_solver": "met_dykstra_corrected",
        "met_mass_fraction": args.met_mass_fraction,
        "met_dustbin_cost": args.met_dustbin_cost,
        "met_iterations": args.met_iterations,
        "tiny_whole_mass": 1e-4, "max_patches": 0,
        "calibrate_episode": parse_bool(args.calibrate),
        "use_superpixels": False, "spix_mode": [],
        "pixel_mean": pixel_mean, "pixel_std": pixel_std,
        "transductive": args.transductive,
    }


def pred_labels_from_logits(logits, support_labels):
    classes = torch.unique(support_labels.detach()).sort().values.to(logits.device)
    return classes[logits.argmax(dim=1)]


def evaluate_pair(args):
    uot = load_runtime_modules()
    uot.DEVICE = args.device
    set_all_seeds(args.seed)
    image_size = int(args.image_size)
    base_backbone, patch_size, default_transform = uot.load_dino_backbone(args.backbone, image_size)
    cand_backbone = copy.deepcopy(base_backbone).to(args.device)
    base_backbone.eval(); cand_backbone.eval()
    ds_specific = uot.build_ds_transforms(image_size)
    transform = ds_specific[uot._dataset_transform_key(args.dataset)] if parse_bool(args.use_specific_trans) else default_transform
    stats = uot.extract_normalize_stats(transform)
    pixel_mean, pixel_std = stats if stats is not None else (None, None)
    test_set = build_dataset(uot, args.dataset, transform, args)
    sampler = uot.TaskSampler(test_set, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, n_tasks=args.n_test_tasks)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=args.n_workers, pin_memory=True, collate_fn=sampler.episodic_collate_fn)

    base_kwargs = model_kwargs(args, patch_size, pixel_mean, pixel_std)
    baseline = uot.OTNet(backbone=base_backbone, **base_kwargs).to(args.device)
    baseline.n_query = args.n_query
    candidate = uot.OTNet(backbone=cand_backbone, **base_kwargs).to(args.device)
    candidate.n_query = args.n_query

    run_dir = Path(args.output_dir) / "runs" / args.run_id
    episode_path = run_dir / "episodes.jsonl"
    base_corr = cand_corr = total = 0
    wins = ties = losses = 0
    episode_idx = 0

    for episode_idx, batch in enumerate(loader):
        si, sl, qi, ql, _ = batch
        si = si.to(args.device, non_blocking=True); sl = sl.to(args.device, non_blocking=True)
        qi = qi.to(args.device, non_blocking=True); ql = ql.to(args.device, non_blocking=True)

        with torch.no_grad():
            baseline.process_support_set(si, sl)
            blog = baseline(qi)
            bpred = pred_labels_from_logits(blog, sl)
            b_corr = int((bpred == ql).sum().item())

        ep_seed = int(args.seed) * 100_000 + int(episode_idx)
        set_all_seeds(ep_seed)
        with torch.no_grad():
            clog, dbg = tta_forward(
                candidate, si, sl, qi,
                K=args.tta_k, aug_strength=args.tta_aug_strength,
                avg_mode=args.tta_avg_mode, device=args.device,
            )
            cpred = pred_labels_from_logits(clog, sl)
            c_corr = int((cpred == ql).sum().item())

        n_q = int(ql.numel())
        base_corr += b_corr; cand_corr += c_corr; total += n_q
        if c_corr > b_corr: wins += 1
        elif c_corr == b_corr: ties += 1
        else: losses += 1

        append_jsonl_atomic(episode_path, {
            "run_id": args.run_id, "episode": episode_idx,
            "baseline_accuracy": b_corr/n_q, "candidate_accuracy": c_corr/n_q,
            "delta_accuracy": (c_corr - b_corr)/n_q,
            "baseline_correct": b_corr, "candidate_correct": c_corr,
            "n_queries": n_q, "tta_debug": dbg,
        })
        if (episode_idx + 1) % 10 == 0:
            cur = 100.0 * (cand_corr - base_corr) / float(total)
            print(f"[tta] ep {episode_idx+1}/{args.n_test_tasks} Δpp={cur:+.3f} W/T/L={wins}/{ties}/{losses}", flush=True)

    return EvalSummary(
        baseline_accuracy=base_corr/total, candidate_accuracy=cand_corr/total,
        delta_accuracy=(cand_corr - base_corr)/total,
        delta_accuracy_pp=100.0*(cand_corr - base_corr)/total,
        baseline_correct=base_corr, candidate_correct=cand_corr,
        total_queries=total, episodes=int(args.n_test_tasks),
        candidate_wins=wins, candidate_ties=ties, candidate_losses=losses,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--backbone", type=str, default="vits16", choices=["vits16","vits8"])
    p.add_argument("--n_way", type=int, default=5); p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=10); p.add_argument("--n_test_tasks", type=int, default=100)
    p.add_argument("--n_workers", type=int, default=4); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--use_specific_trans", type=parse_bool, default=True)
    p.add_argument("--reg_eps", type=float, default=0.01); p.add_argument("--reg_mass", type=float, default=0.4)
    p.add_argument("--alpha_global", type=float, default=0.6)
    p.add_argument("--met_mass_fraction", type=float, default=0.7); p.add_argument("--met_dustbin_cost", type=float, default=0.5)
    p.add_argument("--met_iterations", type=int, default=200); p.add_argument("--calibrate", type=parse_bool, default=True)
    p.add_argument("--transductive", type=str, default="sinkhorn", choices=["sinkhorn","none"])
    p.add_argument("--tta_k", type=int, default=4, help="Number of augmented views (incl. clean view as view 0)")
    p.add_argument("--tta_aug_strength", type=float, default=0.3)
    p.add_argument("--tta_avg_mode", type=str, default="probs", choices=["logits","probs"])
    p.add_argument("--output_dir", type=str, default="experiments/tta_inference")
    p.add_argument("--run_id", type=str, default=""); p.add_argument("--skip_existing", type=parse_bool, default=True)
    args = p.parse_args(argv)
    if not args.run_id:
        args.run_id = f"{args.dataset.replace('-','_')}__tta_K{args.tta_k}_aug{args.tta_aug_strength}_{args.tta_avg_mode}__tasks{args.n_test_tasks}"
    return args


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.output_dir) / "runs" / args.run_id
    status_path = run_dir / "status.json"; result_path = run_dir / "result.json"
    if parse_bool(args.skip_existing) and result_path.exists():
        print(result_path.read_text()); return 0
    write_json(status_path, {"status": "running", "run_id": args.run_id, "argv": sys.argv,
                              "started_at": datetime.now().isoformat(),
                              "hostname": os.uname().nodename, "pid": os.getpid()})
    try:
        started = time.time()
        summary = evaluate_pair(args)
        result = {"status": "completed", "run_id": args.run_id, "summary": asdict(summary),
                  "config": vars(args), "elapsed_seconds": time.time() - started,
                  "completed_at": datetime.now().isoformat()}
        write_json(result_path, result)
        append_jsonl_atomic(Path(args.output_dir) / "results.jsonl", result)
        write_json(status_path, {"status": "completed", "run_id": args.run_id,
                                  "delta_accuracy_pp": summary.delta_accuracy_pp,
                                  "completed_at": datetime.now().isoformat()})
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        failure = {"status": "failed", "run_id": args.run_id, "error": repr(exc),
                   "traceback": traceback.format_exc(), "failed_at": datetime.now().isoformat()}
        write_json(status_path, failure)
        append_jsonl_atomic(Path(args.output_dir) / "failures.jsonl", failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
