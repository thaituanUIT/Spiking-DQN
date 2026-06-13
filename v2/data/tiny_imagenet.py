import os
import random
import urllib.request
import zipfile

import cv2
import numpy as np
from torch.utils.data import Dataset


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_ARCHIVE = "tiny-imagenet-200.zip"
TINY_IMAGENET_DIRNAME = "tiny-imagenet-200"


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader for Active Object Localization.

    Supports:
    - train split using per-class bounding box files
    - val split using val_annotations.txt
    """

    def __init__(
        self,
        root_dir,
        target_class="mixing",
        num_samples=None,
        split="train",
        use_random=False,
        download=True,
    ):
        self.root_dir = root_dir
        self.target_class = target_class
        self.num_samples = num_samples
        self.split = split
        self.use_random = use_random
        self.download = download

        self.dataset_dir = self._resolve_dataset_dir(root_dir)
        self._ensure_available()
        self.words_map = self._load_words()

        self.samples = []
        self._load_data()

    def _resolve_dataset_dir(self, root_dir):
        normalized = os.path.abspath(root_dir)
        if os.path.basename(normalized) == TINY_IMAGENET_DIRNAME:
            return normalized
        return os.path.join(normalized, TINY_IMAGENET_DIRNAME)

    def _ensure_available(self):
        if os.path.isdir(self.dataset_dir):
            return

        if not self.download:
            raise FileNotFoundError(
                f"Tiny ImageNet dataset not found at {self.dataset_dir}. "
                "Create the repo-local dataset folder or enable download."
            )

        parent_dir = os.path.dirname(self.dataset_dir)
        os.makedirs(parent_dir, exist_ok=True)
        archive_path = os.path.join(parent_dir, TINY_IMAGENET_ARCHIVE)

        if not os.path.exists(archive_path):
            print(f"Downloading Tiny ImageNet from {TINY_IMAGENET_URL} ...")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, archive_path)

        print(f"Extracting Tiny ImageNet to {parent_dir} ...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(parent_dir)

    def _load_words(self):
        words_path = os.path.join(self.dataset_dir, "words.txt")
        words_map = {}

        if not os.path.exists(words_path):
            return words_map

        with open(words_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    continue
                wnid, names = parts
                words_map[wnid] = [name.strip().lower() for name in names.split(",")]

        return words_map

    def _matches_target(self, wnid):
        if self.target_class == "mixing":
            return True

        target = self.target_class.strip().lower()
        if target == wnid.lower():
            return True

        return target in self.words_map.get(wnid, [])

    def _load_data(self):
        if self.split == "train":
            self._load_train_samples()
        elif self.split == "val":
            self._load_val_samples()
        else:
            raise ValueError(f"Unsupported Tiny ImageNet split: {self.split}")

        if self.use_random:
            random.seed(42)
            random.shuffle(self.samples)

        if self.num_samples is not None:
            self.samples = self.samples[: self.num_samples]

        self.class_counts = {}
        for sample in self.samples:
            cls = sample["class_name"]
            self.class_counts[cls] = self.class_counts.get(cls, 0) + 1

        print(
            f"Loaded {len(self.samples)} Tiny ImageNet samples "
            f"(split={self.split}, target={self.target_class})."
        )

    def _load_train_samples(self):
        train_dir = os.path.join(self.dataset_dir, "train")
        wnids = sorted(entry for entry in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, entry)))

        for wnid in wnids:
            if not self._matches_target(wnid):
                continue

            boxes_path = os.path.join(train_dir, wnid, f"{wnid}_boxes.txt")
            images_dir = os.path.join(train_dir, wnid, "images")
            class_label = self.words_map.get(wnid, [wnid])[0]

            with open(boxes_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) != 5:
                        continue

                    filename, xmin, ymin, xmax, ymax = parts
                    image_path = os.path.join(images_dir, filename)
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                            "class_name": wnid,
                            "class_label": class_label,
                            "filename": filename,
                        }
                    )

    def _load_val_samples(self):
        val_dir = os.path.join(self.dataset_dir, "val")
        images_dir = os.path.join(val_dir, "images")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")

        with open(annotations_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) != 6:
                    continue

                filename, wnid, xmin, ymin, xmax, ymax = parts
                if not self._matches_target(wnid):
                    continue

                class_label = self.words_map.get(wnid, [wnid])[0]
                self.samples.append(
                    {
                        "image_path": os.path.join(images_dir, filename),
                        "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                        "class_name": wnid,
                        "class_label": class_label,
                        "filename": filename,
                    }
                )

    def get_sample_weights(self):
        return [1.0 / self.class_counts[sample["class_name"]] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {sample['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box = np.array(sample["box"], dtype=np.int32)

        return {
            "image": image,
            "box": box,
            "image_path": sample["image_path"],
            "filename": sample["filename"],
            "class_name": sample["class_name"],
            "class_label": sample["class_label"],
        }
