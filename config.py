from pathlib import Path

APP_TITLE = "Klasifikasi Penyakit Daun Padi"
APP_SUBTITLE = "Deteksi penyakit daun padi berbasis CNN dengan Streamlit"

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")

# Aplikasi akan otomatis mencari model dari beberapa kemungkinan nama file.
# Prioritas pertama adalah hasil training Google Colab:
# models/rice_disease_model.keras + models/class_names.txt
MODEL_CANDIDATES = [
    MODEL_DIR / "rice_disease_model.keras",
    MODEL_DIR / "model_padi.keras",
    Path("model_padi.keras"),
    Path("rice_disease_model.keras"),
]

CLASS_NAMES_CANDIDATES = [
    MODEL_DIR / "class_names.txt",
    MODEL_DIR / "labels.txt",
    Path("labels.txt"),
    Path("class_names.txt"),
]

def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]

MODEL_PATH = first_existing(MODEL_CANDIDATES)
CLASS_NAMES_PATH = first_existing(CLASS_NAMES_CANDIDATES)

HISTORY_PATH = MODEL_DIR / "training_history.csv"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = MODEL_DIR / "classification_report.csv"
DATASET_SUMMARY_PATH = MODEL_DIR / "dataset_summary.csv"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
SEED = 123
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Validasi gambar di luar dataset (Out-of-Distribution / OOD).
# Nilai dapat disesuaikan setelah pengujian menggunakan gambar daun padi dan non-padi.
RICE_MIN_IMAGE_SIDE = 96
RICE_MIN_LEAF_COLOR_RATIO = 0.08
RICE_MIN_CENTER_LEAF_COLOR_RATIO = 0.10
RICE_MIN_CONFIDENCE = 0.40
RICE_MIN_MARGIN = 0.08
RICE_MAX_NORMALIZED_ENTROPY = 0.93
RICE_MIN_TTA_AGREEMENT = 2 / 3

NON_RICE_MESSAGE = (
    "Maaf, gambar tersebut bukan foto penyakit daun padi. "
    "Sistem tidak dapat mengidentifikasi penyakit daun padi."
)
