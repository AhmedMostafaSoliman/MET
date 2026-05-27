"""Official Meta-Dataset episodic evaluation utilities."""

from __future__ import annotations

import math
import os
import resource
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_META_DATASET_CODE_ROOT = "meta-dataset"
DEFAULT_META_DATASET_RECORDS_ROOT = (
    "/home_old/ahmedm04/few_shot_ds/meta-dataset/processed_data"
)

META_DATASET_TEST_DATASETS = [
    "ilsvrc_2012",
    "omniglot",
    "aircraft",
    "cu_birds",
    "dtd",
    "quickdraw",
    "fungi",
    "vgg_flower",
    "traffic_sign",
    "mscoco",
    "mnist",
    "cifar10",
    "cifar100",
]

_META_ALIASES: Dict[str, str] = {
    "ilsvrc_2012": "ilsvrc_2012",
    "ilsvrc2012": "ilsvrc_2012",
    "imagenet": "ilsvrc_2012",
    "metaimagenet": "ilsvrc_2012",
    "metadatasetimagenet": "ilsvrc_2012",
    "aircraft": "aircraft",
    "metaaircraft": "aircraft",
    "metadatasetaircraft": "aircraft",
    "cu_birds": "cu_birds",
    "cubirds": "cu_birds",
    "cu-birds": "cu_birds",
    "metacubirds": "cu_birds",
    "metadatasetcubirds": "cu_birds",
    "dtd": "dtd",
    "metadtd": "dtd",
    "metadatasetdtd": "dtd",
    "fungi": "fungi",
    "metafungi": "fungi",
    "metadatasetfungi": "fungi",
    "omniglot": "omniglot",
    "metaomniglot": "omniglot",
    "metadatasetomniglot": "omniglot",
    "quickdraw": "quickdraw",
    "metaquickdraw": "quickdraw",
    "metadatasetquickdraw": "quickdraw",
    "vgg_flower": "vgg_flower",
    "vggflower": "vgg_flower",
    "vgg-flower": "vgg_flower",
    "metavggflower": "vgg_flower",
    "metadatasetvggflower": "vgg_flower",
    "traffic_sign": "traffic_sign",
    "trafficsign": "traffic_sign",
    "traffic-sign": "traffic_sign",
    "metatrafficsign": "traffic_sign",
    "metadatasettrafficsign": "traffic_sign",
    "mscoco": "mscoco",
    "ms_coco": "mscoco",
    "coco": "mscoco",
    "metamscoco": "mscoco",
    "metacoco": "mscoco",
    "metadatasetmscoco": "mscoco",
    "metadatasetcoco": "mscoco",
    "mnist": "mnist",
    "metamnist": "mnist",
    "metadatasetmnist": "mnist",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "metacifar10": "cifar10",
    "metadatasetcifar10": "cifar10",
    "cifar100": "cifar100",
    "cifar-100": "cifar100",
    "metacifar100": "cifar100",
    "metadatasetcifar100": "cifar100",
}

META_DATASET_TEST_TYPES = [
    "standard",
    "5way",
    "5shot",
    "1shot",
    "5way_5shot",
    "5way_1shot",
]

_FORCED_META_TEST_GIN_BINDINGS = {
    "EpisodeDescriptionConfig.min_ways": 5,
    "EpisodeDescriptionConfig.max_ways_upper_bound": 50,
    "EpisodeDescriptionConfig.max_num_query": 10,
    "EpisodeDescriptionConfig.max_support_set_size": 500,
    "EpisodeDescriptionConfig.max_support_size_contrib_per_class": 100,
    "EpisodeDescriptionConfig.min_log_weight": -0.69314718055994529,
    "EpisodeDescriptionConfig.max_log_weight": 0.69314718055994529,
    "EpisodeDescriptionConfig.ignore_dag_ontology": False,
    "EpisodeDescriptionConfig.ignore_bilevel_ontology": False,
    "EpisodeDescriptionConfig.ignore_hierarchy_probability": 0.0,
    "EpisodeDescriptionConfig.simclr_episode_fraction": 0.0,
    "DataConfig.image_height": 224,
    "DataConfig.shuffle_buffer_size": 1000,
    "DataConfig.read_buffer_size_bytes": 1048576,
    "DataConfig.num_prefetch": 64,
    "ImageDecoder.image_size": 224,
}


def _normalize_alias(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch == "_")


def canonicalize_meta_dataset_name(name: str) -> str:
    key = _normalize_alias(name)
    if key not in _META_ALIASES:
        key = key.replace("_", "")
    if key not in _META_ALIASES:
        raise ValueError(
            f"Unknown Meta-Dataset domain: {name}. Available: {META_DATASET_TEST_DATASETS}"
        )
    return _META_ALIASES[key]


def is_meta_dataset_name(name: str) -> bool:
    key = _normalize_alias(name)
    return key in _META_ALIASES or key.replace("_", "") in _META_ALIASES


def _resolve_project_relative_path(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def remap_episode_labels(
    support_labels: torch.Tensor,
    query_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remap episode labels to contiguous [0..n_way-1] using support labels."""
    class_ids = torch.unique(support_labels).tolist()
    mapping = {int(class_id): idx for idx, class_id in enumerate(class_ids)}
    remapped_support = torch.tensor(
        [mapping[int(x)] for x in support_labels.tolist()],
        device=support_labels.device,
        dtype=torch.long,
    )
    try:
        remapped_query = torch.tensor(
            [mapping[int(x)] for x in query_labels.tolist()],
            device=query_labels.device,
            dtype=torch.long,
        )
    except KeyError as e:
        raise RuntimeError(
            f"Query label {e} is not present in support labels. Episode is invalid."
        ) from e
    return remapped_support, remapped_query


def validate_meta_episode_protocol(md_test_type: str, support_labels: torch.Tensor) -> None:
    constraints = {
        "standard": {"n_way": None, "n_shot": None},
        "5way": {"n_way": 5, "n_shot": None},
        "1shot": {"n_way": None, "n_shot": 1},
        "5shot": {"n_way": None, "n_shot": 5},
        "5way_1shot": {"n_way": 5, "n_shot": 1},
        "5way_5shot": {"n_way": 5, "n_shot": 5},
    }
    if md_test_type not in constraints:
        raise ValueError(f"md_test_type must be one of {META_DATASET_TEST_TYPES}")

    expected = constraints[md_test_type]
    expected_way = expected["n_way"]
    expected_shot = expected["n_shot"]
    if expected_way is None and expected_shot is None:
        return

    support_way = int(torch.unique(support_labels).numel())
    support_counts = torch.bincount(support_labels)
    observed_support_per_class = sorted({int(x) for x in support_counts.tolist()})
    way_ok = expected_way is None or support_way == int(expected_way)
    shot_ok = expected_shot is None or not bool((support_counts != int(expected_shot)).any())
    if way_ok and shot_ok:
        return

    expected_desc = []
    if expected_way is not None:
        expected_desc.append(f"{int(expected_way)}-way")
    if expected_shot is not None:
        expected_desc.append(f"{int(expected_shot)}-shot")
    raise RuntimeError(
        f"{md_test_type} protocol violation: expected {'/'.join(expected_desc)} but got "
        f"{support_way}-way with support-per-class={observed_support_per_class}."
    )


def apply_transform_batch(images: torch.Tensor, transform) -> torch.Tensor:
    """Apply the same torchvision transform pipeline used by folder datasets."""
    from torchvision.transforms.functional import to_pil_image

    if transform is None:
        return torch.clamp((images + 1.0) * 0.5, 0.0, 1.0)

    images_01 = torch.clamp((images + 1.0) * 0.5, 0.0, 1.0)
    transformed = [transform(to_pil_image(img.cpu())) for img in images_01]
    return torch.stack(transformed, dim=0)


def configure_model_for_episode_calibration(model, query_labels: torch.Tensor):
    n_way = int(torch.unique(query_labels).numel())
    counts = torch.bincount(query_labels, minlength=n_way)
    balanced = bool(counts.numel() > 0 and (counts == counts[0]).all())

    if hasattr(model, "n_query") and balanced:
        model.n_query = int(counts[0].item())

    if hasattr(model, "calibrate_episode") and bool(getattr(model, "calibrate_episode")):
        if not balanced:
            raise RuntimeError(
                "Meta-Dataset query labels are not class-balanced. Calibration requires "
                "a fixed n_query per class. Use a balanced protocol or disable calibration."
            )
    return counts, balanced


def release_meta_episode_state(model) -> None:
    for attr in (
        "support_images_cached",
        "support_labels_cached",
        "global_prototypes",
        "last_query_whole",
        "last_query_token_mean",
        "support_whole_embeddings",
        "support_labels",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)
    for attr in ("S_by_class", "b_by_class", "last_patch_posteriors", "support_whole_by_class"):
        if hasattr(model, attr):
            setattr(model, attr, [])


def compute_95_ci(episode_accs: List[float]) -> Optional[float]:
    n = len(episode_accs)
    if n < 2:
        return None
    mean = sum(episode_accs) / n
    var = sum((x - mean) ** 2 for x in episode_accs) / (n - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(n)


def evaluate_meta_dataset_episodes(
    *,
    model,
    dataset_name: str,
    transform,
    n_test_tasks: int,
    device: str,
    meta_dataset_root: Optional[str],
    meta_records_root: Optional[str],
    md_test_type: str,
    repo_root: Optional[Path] = None,
) -> Tuple[float, Optional[float]]:
    from tqdm import tqdm

    sampler = MetaDatasetEpisodeSampler(
        repo_root=Path(repo_root or Path(__file__).resolve().parent),
        meta_dataset_root=meta_dataset_root,
        records_root=meta_records_root,
        test_type=md_test_type,
        image_size=224,
        apply_dino_normalization=False,
    )

    total_correct = 0
    total_predictions = 0
    episode_accs: List[float] = []
    model.eval()
    try:
        with torch.no_grad():
            with tqdm(range(int(n_test_tasks)), total=int(n_test_tasks), leave=False) as tqdm_eval:
                for _ in tqdm_eval:
                    episode = None
                    support_images = None
                    support_labels = None
                    query_images = None
                    query_labels = None
                    try:
                        episode = sampler.sample_test_episode(dataset_name)
                        support_images = apply_transform_batch(
                            episode["context_images"],
                            transform,
                        ).to(device, non_blocking=True)
                        query_images = apply_transform_batch(
                            episode["target_images"],
                            transform,
                        ).to(device, non_blocking=True)

                        support_labels, query_labels = remap_episode_labels(
                            episode["context_labels"],
                            episode["target_labels"],
                        )
                        validate_meta_episode_protocol(md_test_type, support_labels)
                        support_labels = support_labels.to(device, non_blocking=True)
                        query_labels = query_labels.to(device, non_blocking=True)

                        query_counts, query_balanced = configure_model_for_episode_calibration(
                            model,
                            query_labels,
                        )
                        model.process_support_set(support_images, support_labels)
                        predictions = model(query_images).detach()
                        correct = int((predictions.argmax(dim=1) == query_labels).sum().item())
                        total = int(query_labels.numel())

                        total_correct += correct
                        total_predictions += total
                        if total > 0:
                            episode_accs.append(float(correct) / float(total))

                        running_acc = float(total_correct) / float(max(total_predictions, 1))
                        support_way = int(torch.unique(support_labels).numel())
                        support_counts = torch.bincount(support_labels, minlength=max(support_way, 1))
                        support_balanced = bool(
                            support_counts.numel() > 0
                            and (support_counts == support_counts[0]).all()
                        )
                        support_shot = (
                            str(int(support_counts[0].item()))
                            if support_balanced
                            else f"{int(support_counts.min().item())}-{int(support_counts.max().item())}"
                        )
                        tqdm_eval.set_postfix(
                            accuracy=f"{running_acc:.4f}",
                            way_support=support_way,
                            shot_support=support_shot,
                            query_balanced=int(bool(query_balanced)),
                            query_per_class=(
                                int(query_counts[0].item()) if bool(query_balanced) else -1
                            ),
                        )
                    finally:
                        release_meta_episode_state(model)
                        del episode, support_images, support_labels, query_images, query_labels
                        if torch.cuda.is_available() and str(device).startswith("cuda"):
                            torch.cuda.empty_cache()
    finally:
        sampler.close()

    return float(total_correct) / float(max(total_predictions, 1)), compute_95_ci(episode_accs)


def _increase_open_file_limit(target_soft_limit: int = 50000) -> None:
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(max(soft_limit, target_soft_limit), hard_limit)
        if new_soft != soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))
    except Exception:
        pass


def _ensure_tf_estimator_compat(tf) -> None:
    if hasattr(tf, "estimator"):
        return
    tf.estimator = SimpleNamespace(SessionRunHook=type("SessionRunHook", (), {}))


def _ensure_simclr_data_util_compat() -> None:
    try:
        import simclr.data_util  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    simclr_mod = sys.modules.get("simclr")
    if simclr_mod is None:
        simclr_mod = ModuleType("simclr")
        simclr_mod.__path__ = []
        sys.modules["simclr"] = simclr_mod

    data_util_mod = ModuleType("simclr.data_util")

    def _identity(x, *args, **kwargs):
        del args, kwargs
        return x

    data_util_mod.preprocess_for_train = _identity
    data_util_mod.random_blur = _identity
    sys.modules["simclr.data_util"] = data_util_mod
    setattr(simclr_mod, "data_util", data_util_mod)


def _disable_tf_gpu(tf) -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
        return
    except Exception:
        pass
    try:
        tf.config.experimental.set_visible_devices([], "GPU")
    except Exception:
        pass


def _force_meta_testing_gin_bindings(gin) -> None:
    with gin.unlock_config():
        for key, value in _FORCED_META_TEST_GIN_BINDINGS.items():
            gin.bind_parameter(key, value)


class MetaDatasetEpisodeSampler:
    """Episode sampler mirroring the official Meta-Dataset test protocol."""

    def __init__(
        self,
        *,
        repo_root: Path,
        meta_dataset_root: Optional[str],
        records_root: Optional[str],
        test_type: str = "standard",
        image_size: int = 224,
        apply_dino_normalization: bool = False,
    ):
        if test_type not in META_DATASET_TEST_TYPES:
            raise ValueError(f"test_type must be one of {META_DATASET_TEST_TYPES}")

        _increase_open_file_limit(50000)

        self.repo_root = Path(repo_root).resolve()
        self.meta_dataset_root = self._resolve_meta_dataset_root(meta_dataset_root)
        self.records_root = self._resolve_records_root(records_root)
        self.test_type = str(test_type)
        self.image_size = 224
        if int(image_size) != 224:
            print("[warn] Meta-Dataset testing is forced to image_size=224.")
        self.apply_dino_normalization = False
        if bool(apply_dino_normalization):
            print("[warn] apply_dino_normalization is ignored in this evaluation path.")

        if str(self.meta_dataset_root) not in sys.path:
            sys.path.insert(0, str(self.meta_dataset_root))

        try:
            import gin
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'gin-config' required for Meta-Dataset evaluation. "
                "Install it with `python -m pip install gin-config`."
            ) from e

        try:
            import tensorflow as tf
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Missing dependency 'tensorflow' required for Meta-Dataset evaluation. "
                "Install it with `python -m pip install tensorflow`."
            ) from e

        _ensure_tf_estimator_compat(tf)
        _disable_tf_gpu(tf)
        _ensure_simclr_data_util_compat()

        from meta_dataset.data import config
        from meta_dataset.data import decoder as decoder_lib
        from meta_dataset.data import dataset_spec as dataset_spec_lib
        from meta_dataset.data import learning_spec
        from meta_dataset.data import pipeline

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        config_path = self._resolve_optional_meta_gin_config_path()
        if config_path is not None:
            gin.parse_config_file(str(config_path))
        else:
            self._apply_minimal_episode_decoder_bindings(gin=gin, decoder_lib=decoder_lib)
        _force_meta_testing_gin_bindings(gin)

        self._tf = tf
        self._dataset_spec_lib = dataset_spec_lib
        self._learning_spec = learning_spec
        self._pipeline = pipeline
        self._config = config
        session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
        self._session = tf.compat.v1.Session(config=session_config)
        self._next_task_by_dataset: Dict[str, Any] = {}

    @staticmethod
    def _resolve_meta_dataset_root(meta_dataset_root: Optional[str]) -> Path:
        root = meta_dataset_root or os.environ.get("META_DATASET_ROOT") or DEFAULT_META_DATASET_CODE_ROOT
        if not root:
            raise RuntimeError(
                "Meta-Dataset code root is not set. Provide --meta_dataset_root or set $META_DATASET_ROOT."
            )
        resolved = _resolve_project_relative_path(root)
        if not resolved.exists():
            raise FileNotFoundError(f"Meta-Dataset root does not exist: {resolved}")
        return resolved

    @staticmethod
    def _resolve_records_root(records_root: Optional[str]) -> Path:
        root = records_root or os.environ.get("RECORDS") or DEFAULT_META_DATASET_RECORDS_ROOT
        if not root:
            raise RuntimeError(
                "Meta-Dataset records root is not set. Provide --meta_records_root or set $RECORDS."
            )
        resolved = _resolve_project_relative_path(root)
        if not resolved.exists():
            raise FileNotFoundError(f"Meta-Dataset records root does not exist: {resolved}")
        return resolved

    def _resolve_optional_meta_gin_config_path(self) -> Optional[Path]:
        explicit = os.environ.get("META_DATASET_GIN_CONFIG")
        if explicit:
            explicit_path = Path(explicit).resolve()
            if not explicit_path.exists():
                raise FileNotFoundError(
                    f"META_DATASET_GIN_CONFIG is set but file does not exist: {explicit_path}"
                )
            return explicit_path

        candidates = [
            self.repo_root / "DIPA" / "data" / "meta_dataset_config.gin",
            self.repo_root / "data" / "meta_dataset_config.gin",
            self.repo_root
            / "Adaptive-Distribution-Calibration-for-Few-Shot-Learning-with-Hierarchical-Optimal-Transport"
            / "data"
            / "meta_dataset_config.gin",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _apply_minimal_episode_decoder_bindings(gin, decoder_lib) -> None:
        with gin.unlock_config():
            gin.bind_parameter("process_episode.support_decoder", decoder_lib.ImageDecoder())
            gin.bind_parameter("process_episode.query_decoder", decoder_lib.ImageDecoder())

    def _build_episode_description(self):
        c = self._config
        if self.test_type == "standard":
            return c.EpisodeDescriptionConfig(
                None,
                None,
                None,
                min_ways=5,
                max_ways_upper_bound=50,
                max_num_query=10,
                max_support_set_size=500,
                max_support_size_contrib_per_class=100,
                min_log_weight=-0.69314718055994529,
                max_log_weight=0.69314718055994529,
                ignore_dag_ontology=False,
                ignore_bilevel_ontology=False,
                ignore_hierarchy_probability=0.0,
                simclr_episode_fraction=0.0,
            )
        if self.test_type == "5way":
            return c.EpisodeDescriptionConfig(
                None,
                None,
                None,
                min_ways=5,
                max_ways_upper_bound=5,
                max_num_query=10,
            )
        if self.test_type == "1shot":
            return c.EpisodeDescriptionConfig(
                None,
                None,
                None,
                min_ways=5,
                max_ways_upper_bound=50,
                max_num_query=10,
                max_support_size_contrib_per_class=1,
            )
        if self.test_type == "5shot":
            return c.EpisodeDescriptionConfig(
                None,
                5,
                None,
                min_ways=5,
                max_ways_upper_bound=50,
                max_num_query=10,
            )
        if self.test_type == "5way_1shot":
            return c.EpisodeDescriptionConfig(
                None,
                None,
                None,
                min_ways=5,
                max_ways_upper_bound=5,
                max_num_query=10,
                max_support_size_contrib_per_class=1,
            )
        return c.EpisodeDescriptionConfig(
            None,
            5,
            None,
            min_ways=5,
            max_ways_upper_bound=5,
            max_num_query=10,
        )

    def _get_dataset_spec(self, dataset_name: str):
        records_path = self.records_root / dataset_name
        if not records_path.exists():
            raise FileNotFoundError(
                f"Converted records not found for '{dataset_name}' in {records_path}."
            )
        return self._dataset_spec_lib.load_dataset_spec(str(records_path))

    def _init_single_source_test_pipeline(self, dataset_name: str):
        split = self._learning_spec.Split.TEST
        episode_description = self._build_episode_description()
        dataset_spec = self._get_dataset_spec(dataset_name)
        use_bilevel_ontology = dataset_name == "omniglot"
        use_dag_ontology = dataset_name == "ilsvrc_2012"
        one_source = self._pipeline.make_one_source_episode_pipeline(
            dataset_spec=dataset_spec,
            use_dag_ontology=use_dag_ontology,
            use_bilevel_ontology=use_bilevel_ontology,
            split=split,
            episode_descr_config=episode_description,
            image_size=self.image_size,
            shuffle_buffer_size=1000,
        )
        iterator = one_source.make_one_shot_iterator()
        return iterator.get_next()

    @staticmethod
    def _to_torch_episode(episode_tuple):
        return {
            "context_images": torch.from_numpy(episode_tuple[0]).permute(0, 3, 1, 2).float(),
            "context_labels": torch.from_numpy(episode_tuple[1]).long(),
            "target_images": torch.from_numpy(episode_tuple[3]).permute(0, 3, 1, 2).float(),
            "target_labels": torch.from_numpy(episode_tuple[4]).long(),
        }

    def sample_test_episode(self, dataset_name: str):
        canonical_name = canonicalize_meta_dataset_name(dataset_name)
        if canonical_name not in self._next_task_by_dataset:
            self._next_task_by_dataset[canonical_name] = self._init_single_source_test_pipeline(
                canonical_name
            )
        episode = self._session.run(self._next_task_by_dataset[canonical_name])[0]
        return self._to_torch_episode(episode)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
