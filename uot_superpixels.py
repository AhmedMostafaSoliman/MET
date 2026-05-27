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

from datetime import datetime
import os
import math
import argparse
import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch.utils.data as data

try:
    from skimage.segmentation import slic
except ImportError:
    slic = None

# EasyFSL
from easyfsl.methods import FewShotClassifier
from easyfsl.datasets import CUB
from easyfsl.samplers import TaskSampler
from easyfsl.utils import evaluate
from easyfsl.datasets.wrap_few_shot_dataset import WrapFewShotDataset

from data.chestx import ChestX
from data.isic import ISICDataset

# DINO ViT backbone
import dino.vision_transformer as vits
from ot.unbalanced import sinkhorn_unbalanced  # returns transport plan Γ
from ot.bregman import sinkhorn               # used for optional query-class capacity calibration
from utils.wandb import WandbLogger
from met_solver import (
    met_average_cost,
    met_dustbin_sinkhorn,
    met_dykstra_projection,
    met_dykstra_projection_corrected,
)
from meta_dataset_eval import (
    DEFAULT_META_DATASET_CODE_ROOT,
    DEFAULT_META_DATASET_RECORDS_ROOT,
    META_DATASET_TEST_TYPES,
    canonicalize_meta_dataset_name,
    evaluate_meta_dataset_episodes,
    is_meta_dataset_name as is_official_meta_dataset_name,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42  # Constant seed for reproducibility

# ----------------------------
# Seed setting utility
# ----------------------------
def set_seed(seed: int):
    """Set seed for reproducibility across Python, NumPy, and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Simple folder dataset (as in your code)
# ----------------------------
class MiniImageNetDataset(data.Dataset):
    """Loads miniImageNet from a CrossDomainFewShot-style JSON filelist.

    JSON format: {"image_names": ["/abs/path.jpg", ...], "image_labels": [0, 1, ...]}
    Labels are re-mapped to 0-indexed contiguous integers.
    """
    def __init__(self, json_path: str, transform=None):
        import json as _json
        with open(json_path, 'r') as f:
            meta = _json.load(f)
        self.image_names = meta['image_names']
        raw_labels = meta['image_labels']
        unique = sorted(set(raw_labels))
        remap = {old: new for new, old in enumerate(unique)}
        self.targets = [remap[l] for l in raw_labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = Image.open(self.image_names[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]


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


META_DATASET_DEFAULT_ROOT = "/home_old/ahmedm04/few_shot_ds/meta-dataset"
META_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm"}


def _meta_dataset_key(name: str) -> Optional[str]:
    normalized = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "metacifar10": "cifar10",
        "metadatasetcifar10": "cifar10",
        "cifar10": "cifar10",
        "metacifar100": "cifar100",
        "metadatasetcifar100": "cifar100",
        "cifar100": "cifar100",
        "metamnist": "mnist",
        "metadatasetmnist": "mnist",
        "mnist": "mnist",
        "metatrafficsign": "traffic_sign",
        "metatrafficsigns": "traffic_sign",
        "metagtsrb": "traffic_sign",
        "metadatasettrafficsign": "traffic_sign",
        "trafficsign": "traffic_sign",
        "trafficsigns": "traffic_sign",
        "gtsrb": "traffic_sign",
        "metaomniglot": "omniglot",
        "metadatasetomniglot": "omniglot",
        "omniglot": "omniglot",
        "metamscoco": "mscoco",
        "metacoco": "mscoco",
        "metadatasetmscoco": "mscoco",
        "mscoco": "mscoco",
        "coco": "mscoco",
    }
    return aliases.get(normalized)


def _is_meta_dataset(name: str) -> bool:
    return is_official_meta_dataset_name(name)


def _dataset_transform_key(name: str) -> str:
    return "Meta-Dataset" if _is_meta_dataset(name) else name


def _iter_image_files(root: str) -> List[str]:
    paths = []
    for current_root, _, files in os.walk(root):
        for filename in sorted(files):
            if os.path.splitext(filename)[1].lower() in META_IMAGE_EXTENSIONS:
                paths.append(os.path.join(current_root, filename))
    return sorted(paths)


class MetaDatasetFolder(data.Dataset):
    """Small image-folder view over the locally prepared Meta-Dataset subsets."""

    def __init__(
        self,
        root: str = META_DATASET_DEFAULT_ROOT,
        dataset_name: str = "Meta-CIFAR100",
        split: str = "test",
        transform=None,
    ):
        self.root = root
        self.dataset_key = _meta_dataset_key(dataset_name)
        if self.dataset_key is None:
            raise ValueError(f"Unknown Meta-Dataset subset: {dataset_name}")
        self.split = str(split).lower()
        self.transform = transform
        self.data, self.targets = self._load_samples()

    def _split_classes(self, split_name: str) -> Optional[List[str]]:
        split_path = os.path.join(self.root, "splits", f"{split_name}_splits.json")
        if not os.path.exists(split_path):
            return None
        with open(split_path, "r") as f:
            splits = json.load(f)
        classes = splits.get(self.split)
        if classes is None:
            raise ValueError(f"Split '{self.split}' is not available in {split_path}")
        if not classes:
            raise ValueError(f"Split '{self.split}' is empty for Meta-Dataset subset '{self.dataset_key}'")
        return list(classes)

    def _dataset_spec(self, name: str) -> Dict[str, object]:
        spec_path = os.path.join(self.root, "processed_data", name, "dataset_spec.json")
        with open(spec_path, "r") as f:
            return json.load(f)

    def _load_class_folders(
        self,
        base_dir: str,
        class_dirs: Sequence[Tuple[str, str]],
    ) -> Tuple[List[str], List[int]]:
        data = []
        targets = []
        for class_idx, (dir_name, _) in enumerate(class_dirs):
            class_dir = os.path.join(base_dir, dir_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Missing Meta-Dataset class folder: {class_dir}")
            image_paths = _iter_image_files(class_dir)
            if not image_paths:
                raise ValueError(f"No images found in Meta-Dataset class folder: {class_dir}")
            data.extend(image_paths)
            targets.extend([class_idx] * len(image_paths))
        return data, targets

    def _load_regular_folder_subset(self, name: str) -> Tuple[List[str], List[int]]:
        classes = self._split_classes(name)
        base_dir = os.path.join(self.root, "data", name)
        class_dirs = [(class_name, class_name) for class_name in classes]
        return self._load_class_folders(base_dir, class_dirs)

    def _load_traffic_sign(self) -> Tuple[List[str], List[int]]:
        classes = self._split_classes("traffic_sign")
        spec = self._dataset_spec("traffic_sign")
        name_to_id = {class_name: int(class_id) for class_id, class_name in spec["class_names"].items()}
        base_dir = os.path.join(self.root, "data", "GTSRB", "Final_Training", "Images")
        class_dirs = [(f"{name_to_id[class_name]:05d}", class_name) for class_name in classes]
        return self._load_class_folders(base_dir, class_dirs)

    def _load_omniglot(self) -> Tuple[List[str], List[int]]:
        base_dir = os.path.join(self.root, "data", "omniglot", "images_evaluation")
        class_dirs = []
        for alphabet in sorted(os.listdir(base_dir)):
            alphabet_dir = os.path.join(base_dir, alphabet)
            if not os.path.isdir(alphabet_dir):
                continue
            for character in sorted(os.listdir(alphabet_dir)):
                character_dir = os.path.join(alphabet_dir, character)
                if os.path.isdir(character_dir):
                    class_dirs.append((os.path.join(alphabet, character), f"{alphabet}/{character}"))
        if not class_dirs:
            raise ValueError(f"No Omniglot classes found under {base_dir}")
        return self._load_class_folders(base_dir, class_dirs)

    def _load_mscoco(self) -> Tuple[List[str], List[int]]:
        classes = self._split_classes("mscoco")
        annotation_path = os.path.join(self.root, "data", "mscoco", "annotations", "instances_train2017.json")
        image_root = os.path.join(self.root, "data", "mscoco", "train2017")
        with open(annotation_path, "r") as f:
            coco = json.load(f)

        category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
        target_categories = {cat_id for cat_id, name in category_names.items() if name in set(classes)}
        image_names = {image["id"]: image["file_name"] for image in coco["images"]}
        by_category = {cat_id: set() for cat_id in target_categories}
        for ann in coco["annotations"]:
            cat_id = ann["category_id"]
            if cat_id in by_category:
                by_category[cat_id].add(image_names[ann["image_id"]])

        data = []
        targets = []
        ordered_categories = [cat_id for cat_id, name in sorted(category_names.items(), key=lambda item: item[1]) if cat_id in target_categories]
        for class_idx, cat_id in enumerate(ordered_categories):
            image_files = sorted(by_category[cat_id])
            if not image_files:
                raise ValueError(f"No MSCOCO images found for category '{category_names[cat_id]}'")
            data.extend(os.path.join(image_root, filename) for filename in image_files)
            targets.extend([class_idx] * len(image_files))
        return data, targets

    def _load_samples(self) -> Tuple[List[str], List[int]]:
        if self.dataset_key in {"cifar10", "cifar100", "mnist"}:
            return self._load_regular_folder_subset(self.dataset_key)
        if self.dataset_key == "traffic_sign":
            return self._load_traffic_sign()
        if self.dataset_key == "omniglot":
            return self._load_omniglot()
        if self.dataset_key == "mscoco":
            return self._load_mscoco()
        raise ValueError(f"Unsupported Meta-Dataset subset: {self.dataset_key}")

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.data)


class FastWrapFewShotDataset(WrapFewShotDataset):
    """
    Faster Replacement for WrapFewShotDataset that avoids re-loading every image
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


def to_numpy_64(x: torch.Tensor) -> np.ndarray:
    y = x.detach().cpu().numpy()
    if y.dtype != np.float64:
        y = y.astype(np.float64, copy=False)
    return y


def extract_normalize_stats(transform) -> Optional[Tuple[List[float], List[float]]]:
    """
    Recursively search a transform pipeline for the first Normalize op.
    Returns (mean, std) lists if found, else None.
    """
    if transform is None:
        return None
    # torchvision Compose types
    if isinstance(transform, (transforms.Compose, T.Compose)):
        for sub in transform.transforms:
            stats = extract_normalize_stats(sub)
            if stats is not None:
                return stats
        return None
    if isinstance(transform, transforms.Normalize):
        mean = transform.mean.tolist() if isinstance(transform.mean, torch.Tensor) else list(transform.mean)
        std = transform.std.tolist() if isinstance(transform.std, torch.Tensor) else list(transform.std)
        return mean, std
    return None


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
        transport_solver: str = "uot",
        met_mass_fraction: float = 0.6,
        met_dustbin_cost: float = 0.5,
        met_iterations: int = 200,
        tiny_whole_mass: float = 1e-4,
        max_patches: int = 0,            # 0 => use all tokens
        calibrate_episode: bool = True,  # use OT to enforce N_query per class
        use_superpixels: bool = False,
        spix_mode: Optional[Sequence[str]] = None,
        spix_gamma: float = 0.5,
        spix_n_segments: int = 150,
        spix_compactness: float = 10.0,
        spix_mass_mode: str = "area",
        pixel_mean: Optional[Sequence[float]] = None,
        pixel_std: Optional[Sequence[float]] = None,
        # transductive method selection + knobs (used only if calibrate_episode=True)
        transductive: str = "sinkhorn",
        # BD-CSPN
        bd_temp: float = 10.0,
        bd_steps: int = 10,
        bd_lr: float = 0.5,
        # LaplacianShot
        lap_k: int = 10,
        lap_alpha: float = 0.8,
        lap_steps: int = 30,
        # TIM
        tim_temp: float = 10.0,
        tim_steps: int = 20,
        tim_lr: float = 0.05,
        tim_ce_w: float = 1.0,
        tim_marginal_ent_w: float = 1.0,
        tim_cond_ent_w: float = 1.0,
        # PT-MAP
        pt_temp: float = 10.0,
        pt_steps: int = 20,
        pt_momentum: float = 0.5,
        # TFT
        tft_temp: float = 10.0,
        tft_steps: int = 20,
        tft_lr: float = 0.05,
        tft_ce_w: float = 1.0,
        tft_cond_ent_w: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size

        # -- Minimal knobs --
        self.reg_eps = float(reg_eps)        # entropic regularization
        self.reg_mass = float(reg_mass)      # mass KL penalty (unbalancedness)
        self.alpha_global = float(alpha_global)
        self.transport_solver = str(transport_solver).lower()
        valid_transport_solvers = {"uot", "met_dykstra", "met_dykstra_corrected", "met_dustbin"}
        if self.transport_solver not in valid_transport_solvers:
            raise ValueError(
                "transport_solver must be one of: uot, met_dykstra, "
                "met_dykstra_corrected, met_dustbin"
            )
        self.met_mass_fraction = float(met_mass_fraction)
        self.met_dustbin_cost = float(met_dustbin_cost)
        self.met_iterations = int(met_iterations)
        self.tiny_whole_mass = float(tiny_whole_mass)

        self.max_patches = int(max_patches) if max_patches is not None else 0
        self.calibrate_episode = bool(calibrate_episode)
        self.use_superpixels = bool(use_superpixels)

        if self.use_superpixels and slic is None:
            raise RuntimeError(
                "Superpixel modes require scikit-image. Install it or disable --use_superpixels."
            )
        if self.use_superpixels and self.max_patches:
            raise ValueError("max_patches is not supported when superpixels are enabled.")

        if not spix_mode:
            spix_mode = ["mass"] if self.use_superpixels else []
        elif isinstance(spix_mode, str):
            spix_mode = [spix_mode]
        self.spix_modes: Set[str] = {mode.strip().lower() for mode in spix_mode if mode}

        valid_modes = {"mass", "pool", "smooth"}
        invalid = self.spix_modes.difference(valid_modes)
        if invalid:
            raise ValueError(f"Unknown superpixel modes: {sorted(invalid)} (valid: {sorted(valid_modes)})")

        self.spix_gamma = float(spix_gamma)
        self.spix_n_segments = int(spix_n_segments)
        self.spix_compactness = float(spix_compactness)
        self.spix_mass_mode = str(spix_mass_mode).lower()
        if self.spix_mass_mode not in {"area", "equal"}:
            raise ValueError("spix_mass_mode must be 'area' or 'equal'")

        self.pixel_mean = None
        self.pixel_std = None
        if pixel_mean is not None and pixel_std is not None:
            mean = torch.tensor(pixel_mean, dtype=torch.float32)
            std = torch.tensor(pixel_std, dtype=torch.float32)
            if mean.numel() != std.numel():
                raise ValueError("pixel_mean and pixel_std must have the same length")
            self.pixel_mean = mean.view(1, -1, 1, 1)
            self.pixel_std = std.view(1, -1, 1, 1)

        # episode-specific state set in process_support_set(...)
        self.S_by_class: List[torch.Tensor] = []  # per-class support token matrix [Ns_c, D] (normalized)
        self.b_by_class: List[torch.Tensor] = []  # per-class support masses [Ns_c]
        self.global_prototypes: torch.Tensor = None  # [C,D]
        # keep whole-image supports for transductive methods
        self.support_whole_embeddings: Optional[torch.Tensor] = None  # [Ns, D]
        self.support_labels: Optional[torch.Tensor] = None            # [Ns]
        self.support_whole_by_class: List[torch.Tensor] = []          # per-class tensor [n_shot_c, D]

        self.n_way: int = 0
        self.n_query: int = 0  # must be set by outer script for calibration assert

        # debug / analysis artifacts (optional to consume)
        self.last_patch_posteriors: List[torch.Tensor] = []  # per-query [P,C]

        # transductive selection + knobs
        self.transductive = str(transductive).lower()
        # BD-CSPN
        self.bd_temp = float(bd_temp)
        self.bd_steps = int(bd_steps)
        self.bd_lr = float(bd_lr)
        # LaplacianShot
        self.lap_k = int(lap_k)
        self.lap_alpha = float(lap_alpha)
        self.lap_steps = int(lap_steps)
        # TIM
        self.tim_temp = float(tim_temp)
        self.tim_steps = int(tim_steps)
        self.tim_lr = float(tim_lr)
        self.tim_ce_w = float(tim_ce_w)
        self.tim_marginal_ent_w = float(tim_marginal_ent_w)
        self.tim_cond_ent_w = float(tim_cond_ent_w)
        # PT-MAP
        self.pt_temp = float(pt_temp)
        self.pt_steps = int(pt_steps)
        self.pt_momentum = float(pt_momentum)
        # TFT
        self.tft_temp = float(tft_temp)
        self.tft_steps = int(tft_steps)
        self.tft_lr = float(tft_lr)
        self.tft_ce_w = float(tft_ce_w)
        self.tft_cond_ent_w = float(tft_cond_ent_w)

    # ----------------------------
    # Superpixel helpers
    # ----------------------------
    def _superpixel_pool_enabled(self) -> bool:
        return self.use_superpixels and ("pool" in self.spix_modes)

    def _superpixel_mass_enabled(self) -> bool:
        return self.use_superpixels and ("mass" in self.spix_modes)

    def _superpixel_smooth_enabled(self) -> bool:
        return self.use_superpixels and ("smooth" in self.spix_modes)

    def _prepare_images_for_superpixels(self, images: torch.Tensor) -> torch.Tensor:
        imgs = images.detach()
        if imgs.device.type != "cpu":
            imgs = imgs.to("cpu")
        imgs = imgs.float()
        if self.pixel_mean is not None and self.pixel_std is not None:
            mean = self.pixel_mean.to(dtype=imgs.dtype)
            std = self.pixel_std.to(dtype=imgs.dtype)
            imgs = imgs * std + mean
        imgs = imgs.clamp(0.0, 1.0)
        return imgs

    def _run_slic(self, image: torch.Tensor) -> np.ndarray:
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.ascontiguousarray(image_np)
        try:
            labels = slic(
                image_np,
                n_segments=max(2, self.spix_n_segments),
                compactness=self.spix_compactness,
                channel_axis=-1,
                start_label=0,
            )
        except TypeError:
            labels = slic(
                image_np,
                n_segments=max(2, self.spix_n_segments),
                compactness=self.spix_compactness,
                multichannel=True,
                start_label=0,
            )
        return labels.astype(np.int32, copy=False)

    def _labels_to_patch_groups(self, labels: np.ndarray, ph: int, pw: int) -> Tuple[np.ndarray, np.ndarray]:
        H, W = labels.shape
        patch = self.patch_size
        gh, gw = ph, pw
        group_ids = np.empty(gh * gw, dtype=np.int64)
        for r in range(gh):
            for c in range(gw):
                block = labels[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch]
                flat = block.reshape(-1)
                values, counts = np.unique(flat, return_counts=True)
                group_ids[r * gw + c] = values[np.argmax(counts)]
        unique_labels, inverse = np.unique(group_ids, return_inverse=True)
        group_ids = inverse.astype(np.int64, copy=False)
        sizes = np.bincount(group_ids, minlength=unique_labels.shape[0]).astype(np.int64, copy=False)
        return group_ids, sizes

    def _compute_patch_weights(self, group_ids: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        if group_ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._superpixel_mass_enabled():
            denom = np.maximum(sizes[group_ids].astype(np.float32), 1.0)
            weights = (1.0 / denom) ** float(self.spix_gamma)
        else:
            weights = np.ones_like(group_ids, dtype=np.float32)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            weights = np.ones_like(weights, dtype=np.float32)
            weights_sum = weights.sum()
        return (weights / weights_sum).astype(np.float32, copy=False)

    def _compute_segment_mass(self, sizes: np.ndarray) -> np.ndarray:
        if sizes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        if self.spix_mass_mode == "area":
            base = sizes.astype(np.float32)
        else:
            base = np.ones_like(sizes, dtype=np.float32)
        if self._superpixel_mass_enabled():
            denom = np.maximum(sizes.astype(np.float32), 1.0) ** float(self.spix_gamma)
            base = base / denom
        base_sum = base.sum()
        if base_sum <= 0:
            base = np.ones_like(base, dtype=np.float32)
            base_sum = base.sum()
        return (base / base_sum).astype(np.float32, copy=False)

    def _compute_superpixel_info(
        self, images: torch.Tensor, ph: int, pw: int
    ) -> List[Dict[str, torch.Tensor]]:
        imgs = self._prepare_images_for_superpixels(images)
        results: List[Dict[str, torch.Tensor]] = []
        for b in range(imgs.size(0)):
            labels = self._run_slic(imgs[b])
            group_ids_np, sizes_np = self._labels_to_patch_groups(labels, ph, pw)
            patch_weights_np = self._compute_patch_weights(group_ids_np, sizes_np)
            segment_mass_np = self._compute_segment_mass(sizes_np)
            info = {
                "group_ids": torch.from_numpy(group_ids_np),
                "segment_sizes": torch.from_numpy(sizes_np),
                "patch_weights": torch.from_numpy(patch_weights_np),
                "segment_mass": torch.from_numpy(segment_mass_np),
            }
            results.append(info)
        return results

    def _pool_tokens(
        self, tokens: torch.Tensor, info: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        group_ids = info["group_ids"].to(tokens.device, dtype=torch.long)
        K = int(info["segment_sizes"].numel())
        if K == 0:
            return tokens, torch.full((tokens.size(0),), 1.0 / float(tokens.size(0)), device=tokens.device)
        pooled = torch.zeros((K, tokens.size(1)), device=tokens.device, dtype=tokens.dtype)
        pooled.index_add_(0, group_ids, tokens)
        counts = info["segment_sizes"].to(tokens.device, dtype=torch.float32).clamp(min=1.0)
        pooled = pooled / counts.unsqueeze(1)

        mass = info["segment_mass"].to(tokens.device, dtype=torch.float32)
        mass_sum = mass.sum()
        if mass_sum <= 0:
            mass = torch.full((K,), 1.0 / float(K), device=tokens.device, dtype=torch.float32)
        else:
            mass = mass / mass_sum
        return pooled, mass

    def _smooth_posteriors(
        self, post: torch.Tensor, info: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        group_ids = info["group_ids"].to(post.device, dtype=torch.long)
        K = int(info["segment_sizes"].numel())
        if K == 0:
            return post

    # ----------------------------
    # Transductive helpers
    # ----------------------------
    def _row_normalize(self, X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return X / (X.sum(dim=1, keepdim=True) + eps)

    def _cosine_logits(self, Z: torch.Tensor, prototypes: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        return temp * (Z @ prototypes.t())
        sums = torch.zeros((K, post.size(1)), device=post.device, dtype=post.dtype)
        sums.index_add_(0, group_ids, post)
        counts = info["segment_sizes"].to(post.device, dtype=post.dtype).clamp(min=1.0).unsqueeze(1)
        averaged = sums / counts
        return averaged[group_ids]

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
        self.support_whole_embeddings = G
        self.support_labels = support_labels

        classes = torch.unique(support_labels).tolist()
        self.n_way = len(classes)

        self.S_by_class = []
        self.b_by_class = []
        self.support_whole_by_class = []
        whole_by_class = []
        support_spix_info: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.use_superpixels:
            support_spix_info = self._compute_superpixel_info(support_images, ph, pw)

        for c in classes:
            idx = torch.nonzero(support_labels == c, as_tuple=False).squeeze(1)
            assert idx.numel() > 0, "Each class must have at least one support image"
            tokens_c = X[idx]  # [n_shot, P, D]

            per_image_info: List[Dict[str, torch.Tensor]] = []
            if support_spix_info is not None:
                per_image_info = [support_spix_info[i] for i in idx.tolist()]

            if self._superpixel_pool_enabled() and per_image_info:
                pooled_tokens = []
                pooled_masses = []
                for shot_idx, info in enumerate(per_image_info):
                    pooled, seg_mass = self._pool_tokens(tokens_c[shot_idx], info)
                    pooled_tokens.append(pooled)
                    pooled_masses.append(seg_mass)
                parts_tokens = torch.cat(pooled_tokens, dim=0)
                parts_mass = torch.cat(pooled_masses, dim=0)
            else:
                parts_tokens = tokens_c.reshape(-1, D).contiguous()
                if self._superpixel_mass_enabled() and per_image_info:
                    mass_chunks = [
                        info["patch_weights"].to(DEVICE, dtype=torch.float32) for info in per_image_info
                    ]
                    parts_mass = torch.cat(mass_chunks, dim=0)
                else:
                    parts_mass = torch.full(
                        (parts_tokens.size(0),),
                        1.0 / float(parts_tokens.size(0)),
                        device=DEVICE,
                        dtype=torch.float32,
                    )

            parts_tokens = parts_tokens.to(DEVICE)
            parts_tokens = F.normalize(parts_tokens, dim=1, eps=1e-12)
            parts_mass = parts_mass.to(DEVICE, dtype=torch.float32)
            parts_mass_sum = parts_mass.sum()
            if parts_mass_sum > 0:
                parts_mass = parts_mass / parts_mass_sum
            else:
                parts_mass.fill_(1.0 / float(parts_mass.size(0)))
            S_c = parts_tokens

            # append a tiny-mass whole-image token to tie in global semantics (without double-counting)
            whole_c = G[idx]                           # [n_shot, D]
            whole_center = F.normalize(whole_c.mean(dim=0, keepdim=True), dim=1)  # [1,D]
            S_c = torch.cat([S_c, whole_center], dim=0)  # [n_shot*P + 1, D]

            Ns = S_c.size(0)
            b_c = torch.empty((Ns,), device=DEVICE, dtype=torch.float32)
            b_c[:-1] = parts_mass * (1.0 - self.tiny_whole_mass)
            b_c[-1] = self.tiny_whole_mass

            self.S_by_class.append(S_c.to(DEVICE))
            self.b_by_class.append(b_c.to(DEVICE))

            whole_by_class.append(whole_center.squeeze(0))
            self.support_whole_by_class.append(whole_c)

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

        query_spix_info: Optional[List[Dict[str, torch.Tensor]]] = None
        if self.use_superpixels:
            query_spix_info = self._compute_superpixel_info(query_images, ph, pw)

        parts_logits = torch.zeros((Bq, self.n_way), device=DEVICE, dtype=torch.float32)
        evid_per_query: List[torch.Tensor] = []

        for q in range(Bq):
            Qq = Q_tokens[q]
            info_q = query_spix_info[q] if query_spix_info is not None else None

            if self._superpixel_pool_enabled() and info_q is not None:
                Qq, a_q = self._pool_tokens(Qq, info_q)
            else: # TODO try to support both mass and pool together here
                if self._superpixel_mass_enabled() and info_q is not None:
                    a_q = info_q["patch_weights"].to(DEVICE, dtype=torch.float32)
                else:
                    a_q = torch.full(
                        (Qq.size(0),),
                        1.0 / float(Qq.size(0)),
                        device=DEVICE,
                        dtype=torch.float32,
                    )

            Qq = Qq.to(DEVICE)
            Qq = F.normalize(Qq, dim=1, eps=1e-12)
            a_q = a_q.to(DEVICE, dtype=torch.float32)
            a_sum = a_q.sum()
            if a_sum > 0:
                a_q = a_q / a_sum
            else:
                a_q.fill_(1.0 / float(a_q.size(0)))
            a_np = to_numpy_64(a_q)

            evid_q = torch.zeros((Qq.size(0), self.n_way), device=DEVICE, dtype=torch.float32)

            for c in range(self.n_way):
                S_c = self.S_by_class[c]
                b_c = self.b_by_class[c]
                M = cosine_cost(Qq, S_c)

                if self.transport_solver == "uot":
                    Gamma = sinkhorn_unbalanced(
                        a=a_np,
                        b=to_numpy_64(b_c),
                        M=to_numpy_64(M),
                        reg=float(self.reg_eps),
                        reg_m=float(self.reg_mass),
                        numItermax=500,
                        stopThr=1e-6,
                        verbose=False,
                        log=False,
                    )
                    Gamma_t = torch.from_numpy(Gamma).to(DEVICE, dtype=torch.float32)
                    cost = (Gamma_t * M).sum() / (a_q.sum() + 1e-12)
                elif self.transport_solver == "met_dykstra":
                    Gamma_t = met_dykstra_projection(
                        M,
                        a_q,
                        b_c,
                        epsilon=float(self.reg_eps),
                        mass_fraction=float(self.met_mass_fraction),
                        num_iter=int(self.met_iterations),
                    ).to(DEVICE, dtype=torch.float32)
                    cost = met_average_cost(M, Gamma_t)
                elif self.transport_solver == "met_dykstra_corrected":
                    Gamma_t = met_dykstra_projection_corrected(
                        M,
                        a_q,
                        b_c,
                        epsilon=float(self.reg_eps),
                        mass_fraction=float(self.met_mass_fraction),
                        num_iter=int(self.met_iterations),
                    ).to(DEVICE, dtype=torch.float32)
                    cost = met_average_cost(M, Gamma_t)
                else:
                    Gamma_t = met_dustbin_sinkhorn(
                        M,
                        a_q,
                        b_c,
                        epsilon=float(self.reg_eps),
                        query_reject_cost=float(self.met_dustbin_cost),
                        support_reject_cost=float(self.met_dustbin_cost),
                        num_iter=int(self.met_iterations),
                    ).to(DEVICE, dtype=torch.float32)
                    cost = met_average_cost(M, Gamma_t)

                parts_logits[q, c] = -cost

                evid_q[:, c] = Gamma_t.sum(dim=1)

            evid_per_query.append(evid_q)

        posteriors: List[torch.Tensor] = []
        for q, evid in enumerate(evid_per_query):
            row_sums = evid.sum(dim=1, keepdim=True)
            post = evid / torch.clamp(row_sums, min=1e-12)
            empty_rows = row_sums.squeeze(1) <= 0
            if torch.any(empty_rows):
                post[empty_rows] = 1.0 / float(self.n_way)
            if (
                self._superpixel_smooth_enabled()
                and not self._superpixel_pool_enabled()
                and query_spix_info is not None
            ):
                post = self._smooth_posteriors(post, query_spix_info[q])
            posteriors.append(post)
        self.last_patch_posteriors = posteriors

        # global head (pure whole‑image cosine to mean whole embeddings per class)
        global_logits = (Q_whole @ self.global_prototypes.t())  # [Bq, C]

        # blend
        logits = self.alpha_global * global_logits + (1.0 - self.alpha_global) * parts_logits

        # optional transductive refinement OR capacity calibration
        if self.calibrate_episode:
            assert self.n_query > 0 and Bq == (self.n_way * self.n_query), \
                "For calibration, forward must see all queries of the episode (Bq == n_way * n_query)."

            method = self.transductive if self.transductive else "sinkhorn"

            if method == "sinkhorn":
                # Balanced Sinkhorn assignment over queries -> class capacities
                Mqc = -logits  # [Bq, C]
                a_q = torch.full((Bq,), 1.0 / float(Bq), device=DEVICE, dtype=torch.float32)
                b_c = torch.full((self.n_way,), 1.0 / float(self.n_way), device=DEVICE, dtype=torch.float32)

                Gamma_qc = sinkhorn(
                    to_numpy_64(a_q),
                    to_numpy_64(b_c),
                    to_numpy_64(Mqc),
                    reg=float(self.reg_eps),
                    numItermax=500,
                    stopThr=1e-6,
                    verbose=False,
                )
                Gamma_qc_t = torch.from_numpy(Gamma_qc).to(DEVICE, dtype=torch.float32)  # [Bq,C]
                logits = logits + torch.log(Gamma_qc_t + 1e-12)

            elif method == "bdcspn":
                temp = self.bd_temp
                iters = self.bd_steps
                lr = self.bd_lr
                S = self.global_prototypes.clone()
                with torch.no_grad():
                    Q = Q_whole
                    S = F.normalize(S, dim=1)
                for _ in range(iters):
                    Lq = self._cosine_logits(Q, S, temp)
                    Pq = torch.softmax(Lq, dim=1)
                    num = []
                    den = []
                    for c in range(self.n_way):
                        s_sum = self.support_whole_by_class[c].sum(dim=0, keepdim=True)
                        q_sum = (Pq[:, c:c+1] * Q).sum(dim=0, keepdim=True)
                        num.append(s_sum + q_sum)
                        n_shot_c = float(self.support_whole_by_class[c].size(0))
                        den.append(torch.tensor([[n_shot_c]], device=Q.device, dtype=Q.dtype) + Pq[:, c].sum().view(1, 1))
                    new_proto = torch.cat(num, dim=0) / torch.cat(den, dim=0)
                    new_proto = F.normalize(new_proto, dim=1)
                    S = F.normalize((1.0 - lr) * S + lr * new_proto, dim=1)
                logits_ref = self._cosine_logits(Q, S, temp=1.0)
                P = torch.softmax(logits_ref, dim=1)
                logits = logits + torch.log(P.clamp_min(1e-12))

            elif method == "laplacianshot":
                k = self.lap_k
                alpha = self.lap_alpha
                steps = self.lap_steps
                with torch.no_grad():
                    Sqq = torch.clamp(Q_whole @ Q_whole.t(), min=-1.0, max=1.0)
                    Sqq.fill_diagonal_(float('-inf'))
                    _, topi = torch.topk(Sqq, k=k, dim=1)
                    A = torch.zeros_like(Sqq, dtype=torch.bool)
                    A.scatter_(1, topi, True)
                    M = A & A.t()
                    W = torch.maximum(Sqq, torch.zeros_like(Sqq)) * M.float()
                    W.fill_diagonal_(0.0)
                    d = W.sum(dim=1, keepdim=True) + 1e-12
                    Dm12 = 1.0 / torch.sqrt(d)
                    Tsym = Dm12 * W * Dm12.t()
                P0 = torch.softmax(logits, dim=1)
                P_ls = P0.clone()
                for _ in range(steps):
                    P_ls = alpha * (Tsym @ P_ls) + (1.0 - alpha) * P0
                    P_ls = self._row_normalize(P_ls)
                logits = logits + torch.log(P_ls.clamp_min(1e-12))

            elif method == "tim":
                temp = self.tim_temp
                steps = self.tim_steps
                lr = self.tim_lr
                w_ce = self.tim_ce_w
                w_me = self.tim_marginal_ent_w
                w_cey = self.tim_cond_ent_w
                with torch.enable_grad():
                    proto = self.global_prototypes.clone().detach().requires_grad_(True)
                    opt = torch.optim.Adam([proto], lr=lr)
                    Zs = self.support_whole_embeddings
                    ys = self.support_labels
                    Zq = Q_whole
                    for _ in range(steps):
                        Ls = self._cosine_logits(Zs, proto, temp)
                        Lq = self._cosine_logits(Zq, proto, temp)
                        loss_sup = F.cross_entropy(Ls, ys)
                        Pq = torch.softmax(Lq, dim=1)
                        marginal = Pq.mean(dim=0)
                        H_marg = -(marginal * (marginal + 1e-12).log()).sum()
                        H_cond = -(Pq * (Pq + 1e-12).log()).sum(dim=1).mean()
                        loss = w_ce * loss_sup - (w_me * H_marg - w_cey * H_cond)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        with torch.no_grad():
                            proto.data = F.normalize(proto.data, dim=1)
                logits_ref = self._cosine_logits(Zq, proto, temp=1.0)
                P = torch.softmax(logits_ref, dim=1)
                logits = logits + torch.log(P.clamp_min(1e-12))

            elif method == "ptmap":
                temp = self.pt_temp
                steps = self.pt_steps
                eta = self.pt_momentum
                proto = self.global_prototypes.clone()
                Zs, ys = self.support_whole_embeddings, self.support_labels
                Zq = Q_whole
                onehot = F.one_hot(ys, num_classes=self.n_way).float()
                for _ in range(steps):
                    Lq = self._cosine_logits(Zq, proto, temp)
                    Pq = torch.softmax(Lq, dim=1)
                    num = (onehot.t() @ Zs) + (Pq.t() @ Zq)
                    den = onehot.sum(dim=0, keepdim=True).t() + Pq.sum(dim=0, keepdim=True).t()
                    new_proto = F.normalize(num / (den + 1e-12), dim=1)
                    proto = F.normalize((1.0 - eta) * proto + eta * new_proto, dim=1)
                logits_ref = self._cosine_logits(Zq, proto, temp=1.0)
                P = torch.softmax(logits_ref, dim=1)
                logits = logits + torch.log(P.clamp_min(1e-12))

            elif method == "tft":
                temp = self.tft_temp
                steps = self.tft_steps
                lr = self.tft_lr
                w_ce = self.tft_ce_w
                w_cey = self.tft_cond_ent_w
                with torch.enable_grad():
                    proto = self.global_prototypes.clone().detach().requires_grad_(True)
                    opt = torch.optim.Adam([proto], lr=lr)
                    Zs = self.support_whole_embeddings; ys = self.support_labels
                    Zq = Q_whole
                    for _ in range(steps):
                        Ls = self._cosine_logits(Zs, proto, temp)
                        Lq = self._cosine_logits(Zq, proto, temp)
                        loss_sup = F.cross_entropy(Ls, ys)
                        Pq = torch.softmax(Lq, dim=1)
                        H_cond = -(Pq * (Pq + 1e-12).log()).sum(dim=1).mean()
                        loss = w_ce * loss_sup + w_cey * H_cond
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        with torch.no_grad():
                            proto.data = F.normalize(proto.data, dim=1)
                logits_ref = self._cosine_logits(Zq, proto, temp=1.0)
                P = torch.softmax(logits_ref, dim=1)
                logits = logits + torch.log(P.clamp_min(1e-12))

            elif method == "none":
                pass  # no calibration refinement
            else:
                raise ValueError(f"Unknown transductive method: {method}")

        return self.softmax_if_specified(logits)


# ----------------------------
# Dataset‑specific transforms
# ----------------------------
def build_ds_transforms(image_size: int) -> Dict[str, T.Compose]:
    transforms_by_dataset = {
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
        ]),
        # ImageNet-derived datasets use standard ImageNet stats
        "miniImageNet": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "tieredImageNet": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "Meta-Dataset": T.Compose([
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.float() / 255.0),
            transforms.Resize((image_size, image_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    return transforms_by_dataset


# ----------------------------
# Backbone loader
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
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"OTNet_superpixels_{date_time}"
    set_seed(SEED)  # Set seed at the start

    parser = argparse.ArgumentParser(description="OTNet: Few-shot with Unbalanced OT over ViT patches")
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--n_test_tasks', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=12)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--use_specific_trans', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                        default=False)
    parser.add_argument('--backbone', type=str, required=True, choices=["vits16", "vits8"])

    # Hyperparams
    parser.add_argument('--reg_eps', type=float, default=5e-2)
    parser.add_argument('--reg_mass', type=float, default=5e-1)
    parser.add_argument('--alpha_global', type=float, default=0.7)
    parser.add_argument(
        '--transport_solver',
        type=str,
        default='uot',
        choices=['uot', 'met_dykstra', 'met_dykstra_corrected', 'met_dustbin'],
        help=(
            "Patch transport solver: legacy UOT, MET capacity projections, "
            "corrected KL-Dykstra MET, or MET dustbin fallback."
        ),
    )
    parser.add_argument('--met_mass_fraction', type=float, default=0.6, help="Target evidence mass fraction for MET Dykstra.")
    parser.add_argument('--met_dustbin_cost', type=float, default=0.5, help="Rejection cost for MET dustbin solver.")
    parser.add_argument('--met_iterations', type=int, default=200, help="Iterations for MET solvers.")

    # practical toggles
    parser.add_argument('--max_patches', type=int, default=0, help="0 uses all ViT tokens")
    parser.add_argument('--no_calibrate', action='store_true', help="disable episode-capacity calibration")
    # Allow boolean sweep-friendly flags
    parser.add_argument('--use_superpixels', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                        default=False, help="enable superpixel-aware pipeline")
    parser.add_argument(
        '--spix_mode',
        type=str,
        default="mass",
        help="Comma-separated superpixel modes from {mass,pool,smooth}. Only used if --use_superpixels.",
    )
    parser.add_argument('--spix_gamma', type=float, default=0.5, help="Exponent for mass reweighting (gamma in [0,1]).")
    parser.add_argument('--spix_n_segments', type=int, default=150, help="Number of superpixels for SLIC.")
    parser.add_argument('--spix_compactness', type=float, default=10.0, help="SLIC compactness parameter.")
    parser.add_argument(
        '--spix_mass_mode',
        type=str,
        default="area",
        choices=["area", "equal"],
        help="Pooling mass construction: proportional to area or equal per segment.",
    )

    # Calibration control (sweep friendly). If provided, overrides --no_calibrate
    parser.add_argument('--calibrate', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                        default=None, help="Enable/disable episode-capacity calibration (overrides --no_calibrate if set)")

    # W&B logging controls
    parser.add_argument('--use_wandb', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                        default=True, help="Enable/disable Weights & Biases logging")
    parser.add_argument('--wandb_project', type=str, default='MET', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default="leathead_AQ_AM_IO", help='W&B entity (team/user)')
    parser.add_argument('--wandb_offline', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
                        default=False, help='Run W&B in offline mode')
    parser.add_argument('--wandb_name', type=str, default=f'{run_name}', help='Explicit W&B run name')
    parser.add_argument('--wandb_tags', type=str, default='', help='Comma-separated W&B tags')

    parser.add_argument('--mini_imagenet_root', type=str, default='',
                        help='Path to miniImageNet val/ folder (class subdirs). Takes priority over --mini_imagenet_json.')
    parser.add_argument('--mini_imagenet_json', type=str, default='',
                        help='Path to miniImageNet JSON filelist (val.json). Used if --mini_imagenet_root is not set.')
    parser.add_argument('--tiered_imagenet_root', type=str, default='',
                        help='Path to tieredImageNet split folder. Falls back to /home_old/.../val.')
    parser.add_argument(
        '--meta_dataset_root',
        type=str,
        default=DEFAULT_META_DATASET_CODE_ROOT,
        help='Path to the Meta-Dataset code checkout.',
    )
    parser.add_argument(
        '--meta_records_root',
        type=str,
        default=DEFAULT_META_DATASET_RECORDS_ROOT,
        help='Path to converted Meta-Dataset records.',
    )
    parser.add_argument(
        '--md_test_type',
        type=str,
        default='standard',
        choices=META_DATASET_TEST_TYPES,
        help='Official Meta-Dataset test protocol.',
    )
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--seed', type=int, default=SEED, help="Random seed for reproducibility")

    # Transductive options (used only when calibration is enabled)
    parser.add_argument(
        '--transductive',
        type=str,
        default='sinkhorn',
        choices=['sinkhorn', 'none'],
        help="Episode calibration method. Per AGENT.md, only Sinkhorn calibration is exposed.",
    )
    args = parser.parse_args()
    is_meta_dataset = is_official_meta_dataset_name(args.dataset)
    if is_meta_dataset:
        args.dataset = canonicalize_meta_dataset_name(args.dataset)

    # Allow seed override from command line
    if args.seed != SEED:
        set_seed(args.seed)

    DEVICE = args.device
    spix_modes = []
    if args.spix_mode:
        spix_modes = [mode.strip() for mode in args.spix_mode.split(",") if mode.strip()]

    image_size = 224
    ds_specific_transforms = build_ds_transforms(image_size)

    # Backbone
    model, patch_size, datatrans_default = load_dino_backbone(args.backbone, image_size)

    # Dataset
    if args.use_specific_trans:
        datatrans = ds_specific_transforms[_dataset_transform_key(args.dataset)]
        print("LOADED DATASET-SPECIFIC TRANSFORMS")
    else:
        datatrans = datatrans_default

    pixel_mean = pixel_std = None
    stats = extract_normalize_stats(datatrans)
    if stats is not None:
        pixel_mean, pixel_std = stats

    test_loader = None
    if is_meta_dataset:
        test_set = None
    elif args.dataset == "CUB":
        test_set = CUB(split="test", training=False, transform=datatrans)
    elif args.dataset == "Plant-Disease":
        test_set = FastWrapFewShotDataset(MyDataSet('/home_old/ahmedm04/few_shot_ds/Plant-Disease/Plant-Disease', transform=datatrans))
    elif args.dataset == "BCCD_WBC":
        test_set = FastWrapFewShotDataset(MyDataSet('/home_old/ahmedm04/few_shot_ds/BCCD_WBC/BCCD_WBC', transform=datatrans))
    elif args.dataset == "ChestX":
        test_set = FastWrapFewShotDataset(ChestX('/home_old/ahmedm04/few_shot_ds/ChestX', transform=datatrans))
    elif args.dataset == "ISIC":
        test_set = FastWrapFewShotDataset(ISICDataset('/home_old/ahmedm04/few_shot_ds/ISIC2018', transform=datatrans))
    elif args.dataset == "EUROSAT":
        test_set = FastWrapFewShotDataset(MyDataSet('/home_old/ahmedm04/few_shot_ds/EUROSAT/EUROSAT', transform=datatrans))
    elif args.dataset == "HEp":
        test_set = FastWrapFewShotDataset(MyDataSet('/home_old/ahmedm04/few_shot_ds/HEp-Dataset/HEp-Dataset', transform=datatrans))
    elif args.dataset == "miniImageNet":
        # Prefer folder structure (same as tieredImageNet); fall back to JSON filelist.
        _mini_root = args.mini_imagenet_root or "/home_old/ahmedm04/few_shot_ds/mini-imagenet-tools/mini_imagenet_split/val"
        test_set = FastWrapFewShotDataset(MyDataSet(_mini_root, transform=datatrans))
    elif args.dataset == "tieredImageNet":
        _tiered_root = args.tiered_imagenet_root or (
            "/home_old/ahmedm04/few_shot_ds/tiered-imagenet-tools/tiered_imagenet/val"
        )
        test_set = FastWrapFewShotDataset(MyDataSet(_tiered_root, transform=datatrans))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if test_set is not None:
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
    # Calibration decision
    calibrate_episode = (not args.no_calibrate)
    if args.calibrate is not None:
        calibrate_episode = bool(args.calibrate)

    clf = OTNet(
        backbone=model,
        patch_size=patch_size,
        reg_eps=args.reg_eps,
        reg_mass=args.reg_mass,
        alpha_global=args.alpha_global,
        transport_solver=args.transport_solver,
        met_mass_fraction=args.met_mass_fraction,
        met_dustbin_cost=args.met_dustbin_cost,
        met_iterations=args.met_iterations,
        tiny_whole_mass=1e-4,
        max_patches=args.max_patches,
        calibrate_episode=calibrate_episode,
        use_superpixels=args.use_superpixels,
        spix_mode=spix_modes if args.use_superpixels else [],
        spix_gamma=args.spix_gamma,
        spix_n_segments=args.spix_n_segments,
        spix_compactness=args.spix_compactness,
        spix_mass_mode=args.spix_mass_mode,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        transductive=args.transductive,
    ).to(DEVICE)
    clf.n_query = args.n_query  # needed only for calibration assertion

    # Init WandB
    episode_tag = args.md_test_type if is_meta_dataset else f"{args.n_way}way{args.n_shot}shot"
    tags = [t for t in [args.dataset, args.backbone, episode_tag] if t]
    if args.wandb_tags:
        tags.extend([t.strip() for t in args.wandb_tags.split(',') if t.strip()])
    run_name = args.wandb_name or (
        f"{args.dataset}-{args.backbone}-W{args.n_way}S{args.n_shot}-T{args.transductive}-spx{'Y' if args.use_superpixels else 'N'}"
    )

    wandb_config = {
        # Core setup
        'dataset': args.dataset,
        'backbone': args.backbone,
        'patch_size': patch_size,
        'image_size': image_size,
        'n_way': args.n_way,
        'n_shot': args.n_shot,
        'n_query': args.n_query,
        'n_test_tasks': args.n_test_tasks,
        'seed': args.seed,
        # OT hyperparams
        'reg_eps': args.reg_eps,
        'reg_mass': args.reg_mass,
        'alpha_global': args.alpha_global,
        'transport_solver': args.transport_solver,
        'met_mass_fraction': args.met_mass_fraction,
        'met_dustbin_cost': args.met_dustbin_cost,
        'met_iterations': args.met_iterations,
        'tiny_whole_mass': 1e-4,
        'device': DEVICE,
        'max_patches': args.max_patches,
        'meta_dataset_root': args.meta_dataset_root,
        'meta_records_root': args.meta_records_root,
        'md_test_type': args.md_test_type,
        # Calibration + transductive
        'calibrate_episode': calibrate_episode,
        'transductive': args.transductive,
        # Superpixels
        'use_superpixels': args.use_superpixels,
        'spix_mode': spix_modes if args.use_superpixels else [],
        'spix_gamma': args.spix_gamma,
        'spix_n_segments': args.spix_n_segments,
        'spix_compactness': args.spix_compactness,
        'spix_mass_mode': args.spix_mass_mode,
    }
    wb = WandbLogger(
        config=wandb_config,
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="evaluation",
        offline=bool(args.wandb_offline),
        name=run_name,
        tags=tags,
        enabled=bool(args.use_wandb),
    )
    # Only log config explicitly if not running under a sweep
    # (sweep already sets and locks these parameters)
    import wandb
    if wandb.run and not wandb.run.sweep_id:
        wb.log_config(wandb_config)

    # Evaluate
    if is_meta_dataset:
        acc, acc_ci = evaluate_meta_dataset_episodes(
            model=clf,
            dataset_name=args.dataset,
            transform=datatrans,
            n_test_tasks=args.n_test_tasks,
            device=DEVICE,
            meta_dataset_root=args.meta_dataset_root,
            meta_records_root=args.meta_records_root,
            md_test_type=args.md_test_type,
        )
    else:
        acc = evaluate(clf, test_loader, device=DEVICE)
        acc_ci = None
    print(f"Average accuracy : {(100.0 * acc):.2f} %")
    log_payload = {'avg_accuracy': float(acc), 'avg_accuracy_pct': float(round(100.0 * acc, 2))}
    if acc_ci is not None:
        log_payload['avg_accuracy_ci'] = float(acc_ci)
        log_payload['avg_accuracy_ci_pct'] = float(round(100.0 * acc_ci, 2))
    wb.log(log_payload)
    wb.log_summary({
        'avg_accuracy': float(round(acc, 2)),
        'avg_accuracy_pct': float(round(100.0 * acc, 2)),
        'avg_accuracy_ci': None if acc_ci is None else float(acc_ci),
    })

    with open("logs.txt", "a+") as f:
        f.write(
            f"[OTNet] Dataset: {args.dataset}, Backbone: {args.backbone}, Nway: {args.n_way}, Nshot: {args.n_shot}, "
            f"Nquery: {args.n_query}, use_specific_trans: {args.use_specific_trans}, "
            f"reg_eps: {args.reg_eps}, reg_mass: {args.reg_mass}, alpha_global: {args.alpha_global}, "
            f"transport_solver: {args.transport_solver}, met_mass_fraction: {args.met_mass_fraction}, "
            f"met_dustbin_cost: {args.met_dustbin_cost}, met_iterations: {args.met_iterations}, "
            f"max_patches: {args.max_patches}, calibrate: {calibrate_episode}, "
            f"use_superpixels: {args.use_superpixels}, spix_mode: {spix_modes if args.use_superpixels else []}, "
            f"spix_gamma: {args.spix_gamma}, spix_n_segments: {args.spix_n_segments}, "
            f"spix_compactness: {args.spix_compactness}, spix_mass_mode: {args.spix_mass_mode}, "
            f"transductive: {args.transductive}, md_test_type: {args.md_test_type}, "
            f"Accuracy: {acc:.6f}, Note: {args.note}\n"
        )

    wb.finish_run()


if __name__ == "__main__":
    main()
