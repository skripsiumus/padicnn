from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from config import IMAGE_EXTENSIONS, IMAGE_SIZE


def count_images(folder: Path) -> int:
    return sum(1 for file in folder.rglob("*") if file.suffix.lower() in IMAGE_EXTENSIONS)


def find_dataset_root(base_dir: Path) -> Path:
    """Cari folder yang berisi minimal dua subfolder kelas gambar."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {base_dir}")

    folders = [base_dir] + [p for p in base_dir.rglob("*") if p.is_dir()]
    candidates = []

    for folder in folders:
        try:
            child_dirs = [p for p in folder.iterdir() if p.is_dir()]
        except PermissionError:
            continue

        class_dirs = [p for p in child_dirs if count_images(p) > 0]
        if len(class_dirs) >= 2:
            total = sum(count_images(p) for p in class_dirs)
            candidates.append((folder, len(class_dirs), total))

    if not candidates:
        raise ValueError(
            "Struktur dataset belum valid. Pastikan ada minimal 2 folder kelas yang berisi gambar."
        )

    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return candidates[0][0]


def make_dataset_summary(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        total = count_images(class_dir)
        if total > 0:
            rows.append({"kelas": class_dir.name, "jumlah_gambar": total})
    return pd.DataFrame(rows)


def save_class_names(class_names: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for name in class_names:
            file.write(str(name) + "\n")


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def preprocess_image(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
