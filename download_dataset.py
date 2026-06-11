"""
Download dataset Rice Disease dari Kaggle.

Sebelum menjalankan:
1. Install Kaggle: pip install kaggle
2. Login Kaggle dan buat API token dari Account Settings.
3. Letakkan kaggle.json di folder yang diminta Kaggle.

Jalankan:
python download_dataset.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "anshulm257/rice-disease-dataset"
ZIP_NAME = "rice-disease-dataset.zip"
DATASET_DIR = Path("dataset")


def run_command(command: list[str]) -> None:
    print("Menjalankan:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        print("Kaggle CLI belum tersedia. Install dulu dengan:")
        print("pip install kaggle")
        sys.exit(1)

    zip_path = Path(ZIP_NAME)

    try:
        run_command(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", ".", "--force"])
    except subprocess.CalledProcessError:
        print("\nGagal download dari Kaggle.")
        print("Pastikan Anda sudah login Kaggle dan punya file kaggle.json yang benar.")
        sys.exit(1)

    if not zip_path.exists():
        possible_zip = Path("rice-disease-dataset.zip")
        if possible_zip.exists():
            zip_path = possible_zip
        else:
            print("File ZIP dataset tidak ditemukan setelah download.")
            sys.exit(1)

    print(f"Mengekstrak {zip_path} ke {DATASET_DIR}/ ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print("Dataset selesai disiapkan di folder dataset/.")
    print("Langkah berikutnya: python train_model.py")


if __name__ == "__main__":
    main()
