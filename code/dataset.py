"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên để ghép nối với notebook, train.py và eval.py:
    load_split(labels_dir, fold=0)                      -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)              -> torchvision transform
    DeepWeedsDataset[i]                                 -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # chuẩn hoá ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: Union[str, Path], fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_file}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Đảm bảo có đủ cột Filename, Label, Species
    for df in (train_df, val_df, test_df):
        if "Label" in df.columns and "Species" not in df.columns:
            df["Species"] = df["Label"].map(lambda idx: CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"Class_{idx}")

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: Union[str, Path]) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Kiểm tra:
      1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20).
      2. Giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test).
      3. Hợp ba tập phải bằng đúng 17.509 ảnh.
      4. Mọi Filename đều tồn tại trong `images_dir`.
    """
    img_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_samples = n_train + n_val + n_test

    # 1. Phân bố lớp trong từng tập
    per_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        per_class[c_name] = {
            "label": c_idx,
            "train": int((train_df["Label"] == c_idx).sum()),
            "val": int((val_df["Label"] == c_idx).sum()),
            "test": int((test_df["Label"] == c_idx).sum()),
            "total": int((train_df["Label"] == c_idx).sum() + (val_df["Label"] == c_idx).sum() + (test_df["Label"] == c_idx).sum()),
        }

    # 2. Kiểm tra giao giữa các tập
    s_train = set(train_df["Filename"])
    s_val = set(val_df["Filename"])
    s_test = set(test_df["Filename"])

    train_val_overlap = s_train.intersection(s_val)
    train_test_overlap = s_train.intersection(s_test)
    val_test_overlap = s_val.intersection(s_test)

    if len(train_val_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(train_val_overlap)} ảnh trùng giữa train và val!")
    if len(train_test_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(train_test_overlap)} ảnh trùng giữa train và test!")
    if len(val_test_overlap) > 0:
        raise AssertionError(f"Rò rỉ dữ liệu: {len(val_test_overlap)} ảnh trùng giữa val và test!")

    # 3. Kiểm tra hợp 3 tập
    all_filenames = s_train.union(s_val).union(s_test)
    if len(all_filenames) != 17509:
        raise AssertionError(f"Tổng số ảnh duy nhất phải là 17.509, thực tế nhận được {len(all_filenames)}!")

    # 4. Kiểm tra sự tồn tại của file ảnh trong thư mục
    disk_files = set(f.name for f in img_dir.iterdir())
    missing_images = list(all_filenames - disk_files)
    if len(missing_images) > 0:
        raise AssertionError(f"Có {len(missing_images)} ảnh trong CSV không tìm thấy trên đĩa tại {img_dir}!")

    result = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": total_samples,
            "pct_train": round(100.0 * n_train / total_samples, 2),
            "pct_val": round(100.0 * n_val / total_samples, 2),
            "pct_test": round(100.0 * n_test / total_samples, 2),
        },
        "per_class": per_class,
        "overlap": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "total_unique": len(all_filenames),
        "missing_images": len(missing_images),
    }

    print(f"=== KẾT QUẢ KIỂM TRA SPLIT (S1-S4) ===")
    print(f"Số lượng: Train={n_train} ({result['n']['pct_train']}%), Val={n_val} ({result['n']['pct_val']}%), Test={n_test} ({result['n']['pct_test']}%) | Tổng={total_samples}")
    print(f"Giao các tập: train∩val={len(train_val_overlap)}, train∩test={len(train_test_overlap)}, val∩test={len(val_test_overlap)} (Tất cả rỗng - HỢP LỆ)")
    print(f"Hợp 3 tập: {len(all_filenames)} ảnh (Đúng 17.509 ảnh - HỢP LỆ)")
    print(f"Kiểm tra file trên đĩa: Không thiếu ảnh nào trong {img_dir} - HỢP LỆ")

    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> transforms.Compose:
    """Tạo transform cho train hoặc eval/test.

    `aug` (trục B của GUIDE.md mục 3):
      - "basic": RandomResizedCrop(img_size) + RandomHorizontalFlip + Normalize
      - "color": basic + ColorJitter (độ sáng, tương phản, bão hoà, sắc độ)
      - "trivial": TrivialAugmentWide + RandomResizedCrop + RandomHorizontalFlip + Normalize
      - "randaug": RandAugment(2, 9) + RandomResizedCrop + RandomHorizontalFlip + Normalize
    """
    if train:
        if aug == "basic":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "color":
            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "trivial":
            return transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        elif aug == "randaug":
            return transforms.Compose([
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            raise ValueError(f"Không hỗ trợ augmentation: {aug}")
    else:
        # Val / Test: chuẩn hoá xác định, không ngẫu nhiên
        if img_size == 256:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (image_tensor, label: int, filename: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: Union[str, Path], transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].to_numpy(dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fn = self.filenames[i]
        path = self.images_dir / fn
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.labels[i])
        return img, label, fn


def make_loader(df: pd.DataFrame, images_dir: Union[str, Path], transform, batch_size: int,
                train: bool, sampler: Optional[str] = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader tương thích chuẩn."""
    ds = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    sampler_obj = None
    shuffle = train

    if train and sampler == "balanced":
        # Trọng số lấy mẫu tỉ lệ nghịch với số ảnh của lớp
        class_counts = df["Label"].value_counts().to_dict()
        sample_weights = [1.0 / max(1, class_counts[label]) for label in df["Label"]]
        weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
        sampler_obj = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
        shuffle = False

    def _worker_init_fn(worker_id: int):
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    use_cuda = torch.cuda.is_available()
    drop_last = train and (len(ds) > batch_size)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn,
    )
