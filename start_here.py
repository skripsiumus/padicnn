"""
Menu bantuan sederhana untuk pemula.

Jalankan:
python start_here.py
"""

from __future__ import annotations

import subprocess
import sys


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    while True:
        print("\n=== RICE DISEASE CNN - MENU MUDAH ===")
        print("1. Install library")
        print("2. Download dataset Kaggle")
        print("3. Training model CNN")
        print("4. Jalankan web Streamlit")
        print("5. Keluar")

        choice = input("Pilih menu [1-5]: ").strip()

        try:
            if choice == "1":
                run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            elif choice == "2":
                run([sys.executable, "download_dataset.py"])
            elif choice == "3":
                run([sys.executable, "train_model.py"])
            elif choice == "4":
                run([sys.executable, "-m", "streamlit", "run", "app.py"])
            elif choice == "5":
                print("Selesai.")
                break
            else:
                print("Pilihan tidak valid.")
        except subprocess.CalledProcessError as error:
            print(f"Perintah gagal dengan kode: {error.returncode}")
            print("Periksa pesan error di atas, lalu coba lagi.")


if __name__ == "__main__":
    main()
