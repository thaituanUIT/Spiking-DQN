import os

from v2.data.tiny_imagenet import TinyImageNetDataset
from v2.data.voc import VOCDataset
from v2.data.voc_tfds import TFDSVOC2007TestDataset


def get_dataset_root():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dataset",
    )


def get_default_voc_dir():
    return get_dataset_root()


def get_default_tiny_imagenet_dir():
    dataset_root = get_dataset_root()
    preferred_dir = os.path.join(dataset_root, "tiny-imagenet-200")
    legacy_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "IMagenet",
        "tiny-imagenet-200",
    )

    if os.path.isdir(preferred_dir) or not os.path.isdir(legacy_dir):
        return dataset_root
    return os.path.dirname(legacy_dir)


def get_default_weight_path(prefix, dataset_name, target_name, weights_dir):
    if dataset_name == "voc":
        return os.path.join(weights_dir, f"{prefix}_{target_name}.pth")
    return os.path.join(weights_dir, f"{prefix}_{dataset_name}_{target_name}.pth")


def build_train_datasets(
    dataset_name,
    target_class,
    num_samples=None,
    use_random=False,
    validation_mode="none",
    val_ratio=0.2,
):
    if dataset_name == "voc":
        resolved_voc_dir = get_default_voc_dir()
        if validation_mode != "none":
            train_samples = int(num_samples * (1 - val_ratio)) if num_samples else None
            val_samples = int(num_samples * val_ratio) if num_samples else None
            train_dataset = VOCDataset(
                root_dir=resolved_voc_dir,
                target_class=target_class,
                num_samples=train_samples,
                split="train",
                use_random=use_random,
            )
            val_dataset = VOCDataset(
                root_dir=resolved_voc_dir,
                target_class=target_class,
                num_samples=val_samples,
                split="val",
                use_random=use_random,
            )
            return train_dataset, val_dataset

        train_dataset = VOCDataset(
            root_dir=resolved_voc_dir,
            target_class=target_class,
            num_samples=num_samples,
            split="train",
            use_random=use_random,
        )
        return train_dataset, None

    resolved_tiny_imagenet_dir = get_default_tiny_imagenet_dir()
    if validation_mode != "none":
        train_samples = int(num_samples * (1 - val_ratio)) if num_samples else None
        val_samples = int(num_samples * val_ratio) if num_samples else None
        train_dataset = TinyImageNetDataset(
            root_dir=resolved_tiny_imagenet_dir,
            target_class=target_class,
            num_samples=train_samples,
            split="train",
            use_random=use_random,
            download=True,
        )
        val_dataset = TinyImageNetDataset(
            root_dir=resolved_tiny_imagenet_dir,
            target_class=target_class,
            num_samples=val_samples,
            split="val",
            use_random=use_random,
            download=True,
        )
        return train_dataset, val_dataset

    train_dataset = TinyImageNetDataset(
        root_dir=resolved_tiny_imagenet_dir,
        target_class=target_class,
        num_samples=num_samples,
        split="train",
        use_random=use_random,
        download=True,
    )
    return train_dataset, None


def build_eval_dataset(
    dataset_name,
    target_class,
    num_samples=None,
):
    if dataset_name == "voc":
        return TFDSVOC2007TestDataset(target_class=target_class, num_samples=num_samples)

    resolved_tiny_imagenet_dir = get_default_tiny_imagenet_dir()
    return TinyImageNetDataset(
        root_dir=resolved_tiny_imagenet_dir,
        target_class=target_class,
        num_samples=num_samples,
        split="val",
        use_random=False,
        download=True,
    )
