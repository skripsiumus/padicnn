from pathlib import Path

APP_TITLE = "Klasifikasi Penyakit Daun Padi"
APP_SUBTITLE = "Deteksi penyakit daun padi berbasis CNN dengan Streamlit"

DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "rice_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"
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
