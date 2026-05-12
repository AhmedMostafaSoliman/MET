
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mynet_OT_new.py  —  Coherent, low‑knob Optimal Transport few‑shot classifier

Key idea (one strong idea, few knobs):
    Represent each CLASS as the empirical distribution of its support PATCH TOKENS
    (plus a tiny-mass whole‑image token). Classify a QUERY by the (unbalanced)
    OT cost between its patch token distribution and each class distribution.
    Use a single dustbin‑free unbalanced Sinkhorn (no Louvain, no clustering).
    Optionally calibrate the episode with a global OT assignment over queries
    (enforcing N_query per class).

Design choices:
    • Minimal knobs: only three that matter for accuracy
        - reg_eps     (entropic reg for Sinkhorn)
        - reg_mass    (KL penalty for marginal deviations, i.e., unbalancedness)
        - alpha_global (blend with whole‑image head; you asked to keep it)
    • Fails loudly: assertions, no try/except, no fallbacks.
    • Keeps whole‑image embeddings: blended logits or (optionally) injected
      as a tiny‑mass token inside the parts distribution to tie both worlds.

Compatibility:
    • Plugs into EasyFSL like your v2.1 script: we subclass FewShotClassifier
      and implement process_support_set(...) + forward(...).
    • Uses the same DINO ViT backbones and dataset wrappers/paths.

python mynet_OT_v5.py   --dataset EUROSAT --backbone vits16   --n_way 5 --n_shot 5 --n_query 10 --n_test_tasks 100   --reg_eps 0.05 --reg_mass 0.5 --alpha_global 0.5
Average accuracy : 92.44 %

python mynet_OT_v5.py   --dataset ChestX --backbone vits16   --n_way 5 --n_shot 5 --n_query 10 --n_test_tasks 100   --reg_eps 0.05 --reg_mass 0.5 --alpha_global 0.5
Average accuracy : 28.12 %

python mynet_OT_v5.py   --dataset Plant-Disease --backbone vits16  --n_way 5 --n_shot 5 --n_query 10 --n_test_tasks 100   --reg_eps 0.05 --reg_mass 0.5 --alpha_global  0.6                 
Average accuracy : 97.00 %

python mynet_OT_v5.py   --dataset ISIC --backbone vits16   --n_way 5 --n_shot 5 --n_query 10 --n_test_tasks 100   --reg_eps 0.05 --reg_mass 0.5 --alpha_global 0.5
Average accuracy : 48.80 %

python mynet_OT_v5.py --dataset ISIC --backbone vits16 --use_specific_trans --reg_eps 0.02 --reg_mass 0.5 --alpha_global 0.4
Average accuracy : 49.54 %

python mynet_OT_v5.py --dataset ISIC --backbone vits16 --use_specific_trans --reg_eps 0.02 --reg_mass 0.5 --alpha_global 0.3
Average accuracy : 50.40 %

 python mynet_OT_v5.py --dataset ISIC --backbone vits16 --use_specific_trans --reg_eps 0.02 --reg_mass 0.5 --alpha_global 0.4
Average accuracy : 51.84 %

python mynet_OT_v5.py --dataset ISIC --backbone vits16 --use_specific_trans --reg_eps 0.01 --reg_mass 0.5 --alpha_global 0.4 
Average accuracy : 49.50 %


python mynet_OT_v5.py --dataset ISIC --backbone vits16 --use_specific_trans \
>   --reg_eps 0.02 --reg_mass 0.25 --alpha_global 0.4
Average accuracy : 51.54 %
"""

import os
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch.utils.data as data
from torch.func import vmap

# EasyFSL
from easyfsl.methods import FewShotClassifier
from easyfsl.datasets import CUB
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate
from easyfsl.datasets.wrap_few_shot_dataset import WrapFewShotDataset

# Your dataset wrappers (same imports as v2.1)
from data.chestx import ChestX
from data.isic import ISICDataset

# DINO ViT backbone
import dino.vision_transformer as vits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Simple folder dataset (as in your code)
# ----------------------------
class MyDataSet(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data, self.targets = self._load_samples()

    def _load_samples(self):
        data = []
        targets = []
        images_path = self.root
        for class_idx, class_name in enumerate(sorted(os.listdir(images_path))):
            class_imgs_dir = os.path.join(images_path, class_name)
            for sample_name in sorted(os.listdir(class_imgs_dir)):
                sample_path = os.path.join(class_imgs_dir, sample_name)
                data.append(sample_path)
                targets.append(class_idx)
        return data, targets

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        image = Image.open(data).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.data)


class FastWrapFewShotDataset(WrapFewShotDataset):
    """
    Drop-in replacement for WrapFewShotDataset that avoids re-loading every image
    when only labels are required. It reuses any precomputed label storage exposed
    by the underlying dataset (targets, labels, y, etc.) and falls back to the
    original behaviour otherwise.
    """

    def __init__(
        self,
        dataset,
        image_position_in_get_item_output: int = 0,
        label_position_in_get_item_output: int = 1,
        label_attribute_candidates: Tuple[str, ...] = ("targets", "labels", "y"),
    ):
        if image_position_in_get_item_output == label_position_in_get_item_output:
            raise ValueError(
                "image_position_in_get_item_output and label_position_in_get_item_output must be different."
            )
        if (
            image_position_in_get_item_output < 0
            or label_position_in_get_item_output < 0
        ):
            raise ValueError(
                "image_position_in_get_item_output and label_position_in_get_item_output must be positive."
            )

        item_length = len(dataset[0])
        if (
            image_position_in_get_item_output >= item_length
            or label_position_in_get_item_output >= item_length
        ):
            raise ValueError("Specified positions in output are out of range.")

        labels_seq = None
        labels_source = None
        for attr in label_attribute_candidates:
            if hasattr(dataset, attr):
                candidate = getattr(dataset, attr)
                if callable(candidate):
                    candidate = candidate()
                if candidate is not None:
                    labels_seq = candidate
                    labels_source = attr
                    break

        if labels_seq is None:
            super().__init__(dataset, image_position_in_get_item_output, label_position_in_get_item_output)
            return

        labels_list = self._materialize_labels(labels_seq)
        if len(labels_list) != len(dataset):
            super().__init__(dataset, image_position_in_get_item_output, label_position_in_get_item_output)
            return

        self.source_dataset = dataset
        self.labels = labels_list
        self.image_position_in_get_item_output = image_position_in_get_item_output
        self.label_position_in_get_item_output = label_position_in_get_item_output
        self.labels_source = labels_source

    @staticmethod
    def _materialize_labels(labels_seq):
        if isinstance(labels_seq, torch.Tensor):
            labels_seq = labels_seq.detach().cpu()
            labels_list = labels_seq.tolist()
        elif isinstance(labels_seq, np.ndarray):
            labels_list = labels_seq.tolist()
        elif isinstance(labels_seq, list):
            labels_list = labels_seq
        else:
            labels_list = list(labels_seq)

        if not labels_list:
            return labels_list

        first = labels_list[0]
        if isinstance(first, (np.generic, np.integer)):
            return [int(label) for label in labels_list]
        return labels_list


# ----------------------------
# Utilities
# ----------------------------
@torch.no_grad()
def vit_patch_tokens(backbone, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """
    Returns per-image L2-normalized patch tokens as [B,P,D] and (ph, pw)
    """
    feat_all, _, _ = backbone.get_intermediate_feat(images, n=1)
    feat = feat_all[-1]  # [B, 1+HW, D]
    B, L, D = feat.shape
    P = L - 1
    ph = pw = int(math.sqrt(P))
    tokens = feat[:, 1:, :].reshape(B, ph, pw, D).permute(0, 3, 1, 2)  # [B,D,ph,pw]
    tokens = tokens.flatten(2).transpose(1, 2)  # [B, P, D]
    tokens = F.normalize(tokens, dim=2)
    return tokens, ph, pw


@torch.no_grad()
def vit_whole_embeddings(backbone, images: torch.Tensor) -> torch.Tensor:
    """
    Whole-image embeddings [B, D], L2-normalized
    """
    feats = backbone(images)  # [B, D]
    return F.normalize(feats, dim=1)


def cosine_cost(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: [m,D] normalized, B: [n,D] normalized -> returns cost matrix M = 1 - A @ B^T in [0,2]
    """
    S = torch.clamp(A @ B.t(), min=-1.0, max=1.0)
    return 1.0 - S


def sinkhorn_unbalanced_vmap(M: torch.Tensor, a: torch.Tensor, b: torch.Tensor, 
                              reg: float, reg_m: float, numItermax: int = 100) -> torch.Tensor:
    """
    Vmap-compatible unbalanced Sinkhorn without Python control flow.
    
    M: [P, Ns] cost matrix
    a: [P] source marginal  
    b: [Ns] target marginal
    reg: entropic regularization
    reg_m: marginal relaxation (KL penalty)
    
    Returns: [P, Ns] transport plan
    """
    K = torch.exp(-M / reg)
    tau = reg / (reg + reg_m)
    
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    # Fixed iterations (no early stopping for vmap compatibility)
    for _ in range(numItermax):
        Kv = torch.clamp(K @ v, min=1e-9)
        u = torch.pow(a / Kv, tau)
        
        Ktu = torch.clamp(K.t() @ u, min=1e-9)
        v = torch.pow(b / Ktu, tau)
    
    Gamma = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return Gamma


def sinkhorn_balanced_vmap(M: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                            reg: float, numItermax: int = 100) -> torch.Tensor:
    """
    Vmap-compatible balanced Sinkhorn without Python control flow.
    
    M: [m, n] cost matrix
    a: [m] source marginal
    b: [n] target marginal  
    reg: entropic regularization
    
    Returns: [m, n] transport plan
    """
    K = torch.exp(-M / reg)
    
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    # Fixed iterations (no early stopping for vmap compatibility)
    for _ in range(numItermax):
        Kv = torch.clamp(K @ v, min=1e-9)
        u = a / Kv
        
        Ktu = torch.clamp(K.t() @ u, min=1e-9)
        v = b / Ktu
    
    Gamma = (u.unsqueeze(1) * K) * v.unsqueeze(0)
    return Gamma


# ----------------------------
# The model
# ----------------------------
class OTNet(FewShotClassifier):
    """
    Class-conditional Unbalanced OT over patch tokens, with optional episode-level
    query-class capacity calibration, and a simple global (whole-image) head.
    """
    def __init__(
        self,
        backbone,
        patch_size: int,
        reg_eps: float = 5e-2,
        reg_mass: float = 5e-1,
        alpha_global: float = 0.7,
        tiny_whole_mass: float = 1e-4,
        max_patches: int = 0,            # 0 => use all tokens
        calibrate_episode: bool = True,  # use OT to enforce N_query per class
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size

        # -- Minimal knobs --
        self.reg_eps = float(reg_eps)        # entropic regularization
        self.reg_mass = float(reg_mass)      # mass KL penalty (unbalancedness)
        self.alpha_global = float(alpha_global)
        self.tiny_whole_mass = float(tiny_whole_mass)

        self.max_patches = int(max_patches) if max_patches is not None else 0
        self.calibrate_episode = bool(calibrate_episode)

        # episode-specific state set in process_support_set(...)
        self.S_by_class: List[torch.Tensor] = []  # per-class support token matrix [Ns_c, D] (normalized)
        self.b_by_class: List[torch.Tensor] = []  # per-class support masses [Ns_c]
        self.global_prototypes: torch.Tensor = None  # [C,D]

        self.n_way: int = 0
        self.n_query: int = 0  # must be set by outer script for calibration assert

        # debug / analysis artifacts (optional to consume)
        self.last_patch_posteriors: List[torch.Tensor] = []  # per-query [P,C]

    # ----------------------------
    # FewShotClassifier API
    # ----------------------------
    @torch.no_grad()
    def process_support_set(self, support_images: torch.Tensor, support_labels: torch.Tensor):
        """
        Build class-conditional empirical distributions from support patch tokens.
        Also build whole-image global prototypes.
        """
        support_images = support_images.to(DEVICE)
        support_labels = support_labels.to(DEVICE)

        # tokens
        X, ph, pw = vit_patch_tokens(self.backbone, support_images)   # [B,P,D]
        B, P, D = X.shape
        assert P == (ph * pw)

        # optional downsampling of tokens per image to reduce compute (not a "knob" for accuracy)
        if self.max_patches and self.max_patches < P:
            stride = max(1, int(math.floor(P / self.max_patches)))
            idx = torch.arange(0, P, stride, device=X.device)[: self.max_patches]
            X = X[:, idx]  # [B, P', D]
            P = X.size(1)

        # whole embeddings
        G = vit_whole_embeddings(self.backbone, support_images)       # [B, D]

        classes = torch.unique(support_labels).tolist()
        self.n_way = len(classes)

        # per-class concatenate tokens; mass per token = 1/(n_shot * P) so each image contributes equally
        self.S_by_class = []
        self.b_by_class = []
        whole_by_class = []

        for c in classes:
            idx = torch.nonzero(support_labels == c, as_tuple=False).squeeze(1)
            assert idx.numel() > 0, "Each class must have at least one support image"
            S_c = X[idx].reshape(-1, D).contiguous()  # [n_shot*P, D]
            # append a tiny-mass whole-image token to tie in global semantics (without double-counting)
            whole_c = G[idx]                           # [n_shot, D]
            whole_center = F.normalize(whole_c.mean(dim=0, keepdim=True), dim=1)  # [1,D]
            S_c = torch.cat([S_c, whole_center], dim=0)  # [n_shot*P + 1, D]

            Ns = S_c.size(0)
            # equal per-image mass on patches + tiny mass on the whole token
            patch_mass = (1.0 - self.tiny_whole_mass) / float(Ns - 1)
            b_c = torch.full((Ns,), patch_mass, device=DEVICE, dtype=torch.float32)
            b_c[-1] = self.tiny_whole_mass

            self.S_by_class.append(S_c.to(DEVICE))
            self.b_by_class.append(b_c.to(DEVICE))

            # store global prototype (pure global head)
            whole_by_class.append(whole_center.squeeze(0))

        self.global_prototypes = torch.stack(whole_by_class, dim=0)  # [C, D]
        assert self.global_prototypes.shape[0] == self.n_way

    # ----------------------------
    @torch.no_grad()
    def forward(self, query_images: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for all query images in the current episode.
        If calibrate_episode is True, apply an OT-based capacity calibration
        enforcing exactly n_query samples per class (asserted).
        """
        assert len(self.S_by_class) == self.n_way and self.global_prototypes is not None, \
            "Support set not processed. Call process_support_set before forward."

        query_images = query_images.to(DEVICE)

        # tokens and whole
        Q_tokens, ph, pw = vit_patch_tokens(self.backbone, query_images)  # [Bq, P, D]
        Q_whole = vit_whole_embeddings(self.backbone, query_images)       # [Bq, D]

        Bq, Pq, D = Q_tokens.shape

        # optional downsampling consistent with support (same stride heuristic)
        if self.max_patches and self.max_patches < Pq:
            stride = max(1, int(math.floor(Pq / self.max_patches)))
            idx = torch.arange(0, Pq, device=Q_tokens.device)[::stride][: self.max_patches]
            Q_tokens = Q_tokens[:, idx]
            Pq = Q_tokens.size(1)

        # row marginal for queries (uniform over parts per image)
        a = torch.full((Pq,), 1.0 / float(Pq), device=DEVICE, dtype=torch.float32)

        # per-query parts logits via UNBALANCED Sinkhorn OT to each class
        parts_logits = torch.zeros((Bq, self.n_way), device=DEVICE, dtype=torch.float32)

        # collect per-patch evidence to form P(class | query patch)
        evid_per_patch = torch.zeros((Bq, Pq, self.n_way), device=DEVICE, dtype=torch.float32)

        # Vectorize over queries using vmap (still loop over classes)
        for c in range(self.n_way):
            S_c = self.S_by_class[c]     # [Ns_c, D] normalized
            b_c = self.b_by_class[c]     # [Ns_c]
            
            # Compute cost matrices for all queries at once: [Bq, Pq, Ns_c]
            # Using broadcasting: [Bq, Pq, D] @ [D, Ns_c] -> [Bq, Pq, Ns_c]
            S_sim = torch.clamp(Q_tokens @ S_c.t(), min=-1.0, max=1.0)  # [Bq, Pq, Ns_c]
            M_batch = 1.0 - S_sim  # [Bq, Pq, Ns_c]
            
            # Define OT function for single query
            def ot_single_query(M_single):
                return sinkhorn_unbalanced_vmap(
                    M_single, a, b_c,
                    reg=float(self.reg_eps),
                    reg_m=float(self.reg_mass),
                    numItermax=200
                )
            
            # Vectorize over all queries using vmap: [Bq, Pq, Ns_c]
            Gamma_batch = vmap(ot_single_query)(M_batch)  # [Bq, Pq, Ns_c]
            
            # Compute costs for all queries
            costs = (Gamma_batch * M_batch).sum(dim=(1, 2)) / (a.sum() + 1e-12)  # [Bq]
            parts_logits[:, c] = -costs  # higher is better
            
            # Patch-level evidence: sum over support atoms
            evid_per_patch[:, :, c] = Gamma_batch.sum(dim=2)  # [Bq, Pq]

        # normalize per-patch evidences across classes to obtain P(c|patch)
        # (row-wise normalization for each patch; if a row sums to ~0, this will NaN -> we assert)
        row_sums = evid_per_patch.sum(dim=2)  # [Bq,Pq]
        assert torch.all(row_sums > 0), "Unbalanced OT moved zero mass for some query patches; check reg_mass."
        post_per_patch = evid_per_patch / row_sums.unsqueeze(2)  # [Bq,Pq,C]
        self.last_patch_posteriors = [post_per_patch[b] for b in range(Bq)]

        # global head (pure whole‑image cosine to mean whole embeddings per class)
        global_logits = (Q_whole @ self.global_prototypes.t())  # [Bq, C]

        # blend
        logits = self.alpha_global * global_logits + (1.0 - self.alpha_global) * parts_logits

        # optional episode-wise capacity calibration (entropic OT assignment over queries)
        if self.calibrate_episode:
            assert self.n_query > 0 and Bq == (self.n_way * self.n_query), \
                "For calibration, forward must see all queries of the episode (Bq == n_way * n_query)."

            # Cost to assign queries to classes; lower cost => higher affinity
            # Use negative logits (after a temperature implicitly controlled by reg_eps).
            Mqc = -logits  # [Bq, C]
            a_q = torch.full((Bq,), 1.0 / float(Bq), device=DEVICE, dtype=torch.float32)
            b_c = torch.full((self.n_way,), 1.0 / float(self.n_way), device=DEVICE, dtype=torch.float32)  # each class gets exactly Bq/n_way mass

            Gamma_qc_t = sinkhorn_balanced_vmap(
                Mqc,
                a_q,
                b_c,
                reg=float(self.reg_eps),
                numItermax=200,
            )

            # Use assignment mass as a calibration prior: add its log to logits
            # (No extra knob: this is equivalent to one Bayes step with class prior proportional to capacity)
            logits = logits + torch.log(Gamma_qc_t + 1e-12)

        return self.softmax_if_specified(logits)


# ----------------------------
# Dataset‑specific transforms (same stats as v2.1)
# ----------------------------
def build_ds_transforms(image_size: int) -> Dict[str, T.Compose]:
    return {
        "BCCD_WBC": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.6659, 0.6028, 0.7932], std=[0.1221, 0.1698, 0.0543])
        ]),
        "Plant-Disease": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.4662, 0.4888, 0.4101], std=[0.1707, 0.1438, 0.1875])
        ]),
        "EUROSAT": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.3444, 0.3803, 0.4078], std=[0.0884, 0.0621, 0.0521])
        ]),
        "ChestX": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.4920, 0.4920, 0.4920], std=[0.2288, 0.2288, 0.2288])
        ]),
        "ISIC": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0891, 0.1179, 0.1325])
        ]),
        "HEp": T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.7940, 0.7940, 0.7940], std=[0.1920, 0.1920, 0.1920]),
            T.Resize(size=(image_size, image_size)),
        ])
    }


# ----------------------------
# Backbone loader (same URLs as before)
# ----------------------------
def load_dino_backbone(arch_name: str, image_size: int):
    if arch_name == "vits16":
        patch_size = 16
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        arch = "vit_small"
    elif arch_name == "vits8":
        patch_size = 8
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        arch = "vit_small"
    else:
        raise ValueError("backbone must be 'vits16' or 'vits8'")

    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)

    datatrans = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return model, patch_size, datatrans


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="OTNet: Few-shot with Unbalanced OT over ViT patches")
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--n_test_tasks', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--use_specific_trans', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--backbone', type=str, required=True, choices=["vits16", "vits8"])

    # minimal knobs
    parser.add_argument('--reg_eps', type=float, default=5e-2)
    parser.add_argument('--reg_mass', type=float, default=5e-1)
    parser.add_argument('--alpha_global', type=float, default=0.7)

    # practical toggles (not accuracy knobs)
    parser.add_argument('--max_patches', type=int, default=0, help="0 uses all ViT tokens")
    parser.add_argument('--no_calibrate', action='store_true', help="disable episode-capacity calibration")

    parser.add_argument('--note', type=str, default="")
    args = parser.parse_args()

    DEVICE = args.device

    image_size = 224
    ds_specific_transforms = build_ds_transforms(image_size)

    # Backbone
    model, patch_size, datatrans_default = load_dino_backbone(args.backbone, image_size)

    # Dataset
    if args.use_specific_trans:
        datatrans = ds_specific_transforms[args.dataset]
        print("LOADED DATASET-SPECIFIC TRANSFORMS")
    else:
        datatrans = datatrans_default

    if args.dataset == "CUB":
        test_set = CUB(split="test", training=False, transform=datatrans)
    elif args.dataset == "Plant-Disease":
        test_set = FastWrapFewShotDataset(MyDataSet('/home/ahmedm04/projects/distill_part_whole/datasets/Plant-Disease/Plant-Disease', transform=datatrans))
    elif args.dataset == "BCCD_WBC":
        test_set = FastWrapFewShotDataset(MyDataSet('/home/ahmedm04/projects/distill_part_whole/datasets/BCCD_WBC/BCCD_WBC', transform=datatrans))
    elif args.dataset == "ChestX":
        test_set = FastWrapFewShotDataset(ChestX('/home/ahmedm04/projects/DINOSEG/datasets/ChestX', transform=datatrans))
    elif args.dataset == "ISIC":
        test_set = FastWrapFewShotDataset(ISICDataset('/home/ahmedm04/projects/DINOSEG/datasets/ISIC2018', transform=datatrans))
    elif args.dataset == "EUROSAT":
        test_set = FastWrapFewShotDataset(MyDataSet('/home/ahmedm04/projects/distill_part_whole/datasets/EUROSAT/EUROSAT', transform=datatrans))
    elif args.dataset == "HEp":
        test_set = FastWrapFewShotDataset(MyDataSet('/home/ahmedm04/projects/distill_part_whole/datasets/HEp-Dataset/HEp-Dataset', transform=datatrans))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    test_sampler = TaskSampler(
        test_set, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query, n_tasks=args.n_test_tasks
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=args.n_workers,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # Model
    clf = OTNet(
        backbone=model,
        patch_size=patch_size,
        reg_eps=args.reg_eps,
        reg_mass=args.reg_mass,
        alpha_global=args.alpha_global,
        tiny_whole_mass=1e-4,
        max_patches=args.max_patches,
        calibrate_episode=(not args.no_calibrate),
    ).to(DEVICE)
    clf.n_query = args.n_query  # needed only for calibration assertion

    # Evaluate
    acc = evaluate(clf, test_loader, device=DEVICE)
    print(f"Average accuracy : {(100.0 * acc):.2f} %")

    with open("logs.txt", "a+") as f:
        f.write(
            f"[OTNet] Dataset: {args.dataset}, Backbone: {args.backbone}, Nway: {args.n_way}, Nshot: {args.n_shot}, "
            f"Nquery: {args.n_query}, use_specific_trans: {args.use_specific_trans}, "
            f"reg_eps: {args.reg_eps}, reg_mass: {args.reg_mass}, alpha_global: {args.alpha_global}, "
            f"max_patches: {args.max_patches}, calibrate: {not args.no_calibrate}, "
            f"Accuracy: {acc:.6f}, Note: {args.note}\n"
        )


if __name__ == "__main__":
    main()
