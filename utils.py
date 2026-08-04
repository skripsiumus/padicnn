from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps

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


def _to_rgb_array(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Ubah gambar menjadi array RGB uint8 berukuran seragam."""
    rgb_image = image.convert("RGB").resize((image_size, image_size))
    return np.asarray(rgb_image, dtype=np.uint8)


def analyze_leaf_visual(image: Image.Image, image_size: int = IMAGE_SIZE) -> dict:
    """
    Analisis ringan untuk menyaring gambar yang jelas bukan foto daun.

    Filter ini tidak menggantikan model khusus kelas ``bukan daun padi``.
    Tujuannya adalah mencegah foto polos, dokumen, wajah, kendaraan, atau objek
    lain langsung dipaksa masuk ke salah satu kelas penyakit.
    """
    width, height = image.size
    rgb = _to_rgb_array(image, image_size)
    hsv = np.asarray(
        Image.fromarray(rgb, mode="RGB").convert("HSV"),
        dtype=np.uint8,
    )

    hue = hsv[..., 0].astype(np.float32)
    saturation = hsv[..., 1].astype(np.float32)
    value = hsv[..., 2].astype(np.float32)

    # Rentang dibuat cukup luas agar daun sehat, menguning, bercak cokelat,
    # dan daun yang mengering tetap dapat lolos sebagai kandidat foto daun.
    green_mask = (hue >= 35) & (hue <= 125) & (saturation >= 35) & (value >= 25)
    yellow_brown_mask = (
        (((hue >= 8) & (hue < 35)) | (hue >= 245))
        & (saturation >= 35)
        & (value >= 20)
    )
    leaf_color_mask = green_mask | yellow_brown_mask

    margin = max(1, int(image_size * 0.18))
    center_mask = leaf_color_mask[margin:-margin, margin:-margin]

    gray = rgb.astype(np.float32).mean(axis=2)
    horizontal_change = np.abs(np.diff(gray, axis=1)).mean()
    vertical_change = np.abs(np.diff(gray, axis=0)).mean()

    return {
        "width": int(width),
        "height": int(height),
        "min_side": int(min(width, height)),
        "aspect_ratio": float(max(width, height) / max(1, min(width, height))),
        "mean_brightness": float(value.mean()),
        "color_std": float(rgb.astype(np.float32).std()),
        "saturated_ratio": float((saturation >= 35).mean()),
        "leaf_color_ratio": float(leaf_color_mask.mean()),
        "center_leaf_color_ratio": float(center_mask.mean()) if center_mask.size else 0.0,
        "texture_score": float((horizontal_change + vertical_change) / 2.0),
    }


def build_prediction_batch(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Membuat tiga variasi ringan (TTA) untuk mengukur kestabilan prediksi.

    Gambar daun padi yang valid umumnya memberikan prediksi yang relatif stabil
    ketika dicerminkan atau sedikit diubah pencahayaannya.
    """
    rgb = image.convert("RGB")
    variants = [
        rgb,
        ImageOps.mirror(rgb),
        ImageEnhance.Brightness(rgb).enhance(1.08),
    ]
    arrays = []
    for variant in variants:
        resized = variant.resize((image_size, image_size))
        arrays.append(tf.keras.utils.img_to_array(resized))
    return np.stack(arrays, axis=0)


def normalized_entropy(probabilities: np.ndarray) -> float:
    """Entropi 0-1; semakin mendekati 1 berarti model semakin ragu."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-9, 1.0)
    if probabilities.size <= 1:
        return 0.0
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(probabilities.size))

