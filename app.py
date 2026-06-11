"""
Web Streamlit klasifikasi penyakit daun padi menggunakan CNN.

Jalankan:
streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_SUMMARY_PATH,
    HISTORY_PATH,
    IMAGE_SIZE,
    MODEL_INFO_PATH,
    MODEL_PATH,
)
from utils import load_class_names, load_json, preprocess_image
from disease_solutions import (
    get_all_solution_rows,
    get_registered_solution_names,
    get_registered_solution_rows,
    get_solution_for_class,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, #E8F5E9 0%, #F9FFF9 100%);
        border: 1px solid #C8E6C9;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin-bottom: 0.25rem;
        color: #1B5E20;
    }
    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .status-ok {
        padding: 0.7rem 0.85rem;
        border-radius: 0.8rem;
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        color: #1B5E20;
        font-weight: 600;
    }
    .status-bad {
        padding: 0.7rem 0.85rem;
        border-radius: 0.8rem;
        background: #FFF3E0;
        border: 1px solid #FFCC80;
        color: #E65100;
        font-weight: 600;
    }
    .small-note {
        color: #616161;
        font-size: 0.92rem;
    }
    .solution-card {
        padding: 1.1rem 1.2rem;
        border-radius: 1rem;
        background: #F1F8E9;
        border: 1px solid #C5E1A5;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .solution-title {
        color: #33691E;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .solution-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #DCEDC8;
        border: 1px solid #AED581;
        color: #33691E;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_safely():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as error:
        st.error(f"Model ditemukan, tetapi gagal dibaca: {error}")
        return None


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_dataset_summary() -> pd.DataFrame:
    if DATASET_SUMMARY_PATH.exists():
        return pd.read_csv(DATASET_SUMMARY_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report() -> pd.DataFrame:
    if CLASSIFICATION_REPORT_PATH.exists():
        return pd.read_csv(CLASSIFICATION_REPORT_PATH, index_col=0)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_confusion_matrix() -> pd.DataFrame:
    if CONFUSION_MATRIX_PATH.exists():
        return pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
    return pd.DataFrame()


def predict_image(model: tf.keras.Model, image: Image.Image, class_names: list[str]) -> tuple[str, float, pd.DataFrame]:
    image_array = preprocess_image(image, IMAGE_SIZE)
    probabilities = model.predict(image_array, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Jumlah output model tidak sama dengan jumlah class_names.txt. "
            "Coba training ulang model."
        )

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    result_df = pd.DataFrame(
        {
            "Kelas": class_names,
            "Probabilitas": probabilities,
            "Probabilitas (%)": probabilities * 100,
        }
    ).sort_values("Probabilitas (%)", ascending=False)

    return predicted_class, confidence, result_df


def plot_prediction(result_df: pd.DataFrame):
    sorted_df = result_df.sort_values("Probabilitas (%)", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(sorted_df["Kelas"], sorted_df["Probabilitas (%)"])
    ax.set_xlabel("Probabilitas (%)")
    ax.set_ylabel("Kelas")
    ax.set_title("Grafik Confidence Tiap Kelas")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25)

    for index, value in enumerate(sorted_df["Probabilitas (%)"]):
        ax.text(min(value + 1, 99), index, f"{value:.1f}%", va="center")

    fig.tight_layout()
    return fig


def render_solution_box(predicted_class: str):
    solution = get_solution_for_class(predicted_class)

    st.markdown("### Solusi Penanganan")
    st.markdown(
        f"""
        <div class="solution-card">
            <div class="solution-title">{solution['nama_tampilan']}</div>
            <span class="solution-badge">{solution['kategori']}</span>
            <p>{solution['ringkasan']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_gejala, col_tangani, col_cegah = st.columns(3, gap="large")

    with col_gejala:
        st.markdown("#### Gejala Umum")
        for item in solution["gejala"]:
            st.markdown(f"- {item}")

    with col_tangani:
        st.markdown("#### Cara Menangani")
        for item in solution["penanganan"]:
            st.markdown(f"- {item}")

    with col_cegah:
        st.markdown("#### Pencegahan")
        for item in solution["pencegahan"]:
            st.markdown(f"- {item}")

    st.caption(
        "Catatan: rekomendasi ini bersifat umum. Untuk penggunaan pestisida/fungisida/bakterisida, "
        "ikuti label produk dan arahan penyuluh pertanian setempat."
    )


def render_solution_detail(solution: dict):
    st.markdown(
        f"""
        <div class="solution-card">
            <div class="solution-title">{solution['nama_tampilan']}</div>
            <span class="solution-badge">{solution['kategori']}</span>
            <p>{solution['ringkasan']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_gejala, tab_tangani, tab_cegah = st.tabs(["Gejala", "Cara Menangani", "Pencegahan"])
    with tab_gejala:
        for item in solution["gejala"]:
            st.markdown(f"- {item}")
    with tab_tangani:
        for item in solution["penanganan"]:
            st.markdown(f"- {item}")
    with tab_cegah:
        for item in solution["pencegahan"]:
            st.markdown(f"- {item}")


def render_solutions_page():
    st.subheader("Solusi Setiap Penyakit Daun Padi")
    st.write(
        "Halaman ini berisi gejala, cara menangani, dan pencegahan untuk penyakit daun padi "
        "yang tersedia pada aplikasi. Menu ini dapat dibuka tanpa harus melakukan prediksi terlebih dahulu."
    )

    registered_rows = get_registered_solution_rows()
    st.dataframe(pd.DataFrame(registered_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    selected_solution = st.selectbox(
        "Pilih penyakit untuk melihat detail solusi",
        get_registered_solution_names(),
    )
    render_solution_detail(get_solution_for_class(selected_solution))

    st.markdown("---")
    st.markdown("### Semua Solusi")
    for disease_name in get_registered_solution_names():
        solution = get_solution_for_class(disease_name)
        with st.expander(solution["nama_tampilan"]):
            render_solution_detail(solution)

    st.caption(
        "Catatan: rekomendasi ini bersifat umum. Untuk penggunaan pestisida, fungisida, atau bakterisida, "
        "ikuti label produk dan arahan penyuluh pertanian setempat."
    )


def plot_training(history_df: pd.DataFrame, metric: str, val_metric: str, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history_df[metric], marker="o", label="Training")
    ax.plot(history_df[val_metric], marker="o", label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = cm_df.values
    image = ax.imshow(matrix)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def render_commands():
    st.code(
        """# 1. Install library
pip install -r requirements.txt

# 2. Download dataset Kaggle
python download_dataset.py

# 3. Training model CNN
python train_model.py

# 4. Jalankan web
streamlit run app.py""",
        language="bash",
    )


def render_header():
    st.markdown(
        f"""
        <div class="hero">
            <h1>🌾 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <p class="small-note">Upload gambar daun padi, lalu sistem menampilkan hasil prediksi dan grafik confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_missing():
    st.warning("Model belum tersedia. Aplikasi tetap bisa dibuka, tetapi prediksi belum bisa dijalankan.")
    st.markdown("Ikuti langkah berikut untuk membuat model:")
    render_commands()


model = load_model_safely()
class_names = load_class_names(CLASS_NAMES_PATH)
model_info = load_json(MODEL_INFO_PATH)

render_header()

with st.sidebar:
    st.image("https://img.icons8.com/color/96/rice-bowl.png", width=72)
    st.title("Menu")
    page = st.radio(
        "Pilih halaman",
        ["🏠 Beranda", "🔍 Prediksi", "💊 Solusi Penyakit", "📊 Grafik & Evaluasi", "📁 Dataset", "📘 Panduan"],
    )

    st.markdown("---")
    st.subheader("Status")
    if model is not None and class_names:
        st.markdown('<div class="status-ok">Model siap digunakan</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bad">Model belum dilatih</div>', unsafe_allow_html=True)

    if model_info:
        st.caption(f"Model: {model_info.get('model_type', '-')}")
        st.caption(f"Jumlah kelas: {model_info.get('num_classes', '-')}")
        accuracy = model_info.get("accuracy")
        if isinstance(accuracy, (int, float)):
            st.caption(f"Akurasi validasi: {accuracy * 100:.2f}%")

    if class_names:
        with st.expander("Daftar kelas"):
            for i, name in enumerate(class_names, start=1):
                st.write(f"{i}. {name}")


if page == "🏠 Beranda":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="soft-card"><h3>1. Upload</h3><p>Masukkan foto daun padi format JPG, PNG, BMP, atau WEBP.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="soft-card"><h3>2. Prediksi CNN</h3><p>Model membaca pola visual dari gambar daun padi.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="soft-card"><h3>3. Solusi</h3><p>Aplikasi menampilkan gejala, cara menangani, dan pencegahan penyakit daun padi.</p></div>', unsafe_allow_html=True)

    st.subheader("Mulai cepat")
    if model is None or not class_names:
        render_model_missing()
    else:
        st.success("Model sudah siap. Buka halaman Prediksi untuk menguji gambar daun padi.")

    st.info(
        "Aplikasi ini dibuat untuk pembelajaran dan prototipe. Hasil prediksi tidak menggantikan pemeriksaan langsung oleh ahli pertanian."
    )

elif page == "🔍 Prediksi":
    st.subheader("Upload Gambar Daun Padi")

    if model is None or not class_names:
        render_model_missing()
    else:
        uploaded_file = st.file_uploader(
            "Pilih gambar daun padi",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded_file is None:
            st.info("Silakan upload gambar terlebih dahulu.")
        else:
            image = Image.open(uploaded_file)

            left, right = st.columns([1, 1.1], gap="large")
            with left:
                st.image(image, caption="Gambar yang diunggah", use_container_width=True)

            with right:
                with st.spinner("Menganalisis gambar..."):
                    predicted_class, confidence, result_df = predict_image(model, image, class_names)

                st.markdown("### Hasil Prediksi")
                st.success(f"Penyakit terdeteksi: **{predicted_class}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                if confidence < 0.60:
                    st.warning(
                        "Confidence masih rendah. Coba gunakan gambar yang lebih jelas, fokus pada daun, dan pencahayaan cukup."
                    )
                elif confidence < 0.80:
                    st.info("Confidence cukup, tetapi tetap cek kondisi daun secara langsung.")
                else:
                    st.info("Confidence tinggi untuk kelas prediksi teratas.")

                display_df = result_df.copy()
                display_df["Probabilitas (%)"] = display_df["Probabilitas (%)"].map(lambda x: f"{x:.2f}%")
                st.dataframe(display_df[["Kelas", "Probabilitas (%)"]], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Grafik Hasil Klasifikasi")
            st.pyplot(plot_prediction(result_df), use_container_width=True)

            st.markdown("---")
            render_solution_box(predicted_class)

elif page == "💊 Solusi Penyakit":
    render_solutions_page()

elif page == "📊 Grafik & Evaluasi":
    st.subheader("Grafik Training dan Evaluasi Model")

    history_df = load_history()
    report_df = load_report()
    cm_df = load_confusion_matrix()

    if history_df.empty and report_df.empty and cm_df.empty:
        st.warning("Data evaluasi belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
    else:
        if model_info:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Jumlah kelas", model_info.get("num_classes", "-"))
            m2.metric("Jumlah gambar", model_info.get("num_images", "-"))
            acc = model_info.get("accuracy")
            f1 = model_info.get("macro_f1")
            m3.metric("Accuracy", f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else "-")
            m4.metric("Macro F1", f"{f1 * 100:.2f}%" if isinstance(f1, (int, float)) else "-")

        tab1, tab2, tab3 = st.tabs(["Grafik Training", "Classification Report", "Confusion Matrix"])

        with tab1:
            if history_df.empty:
                st.info("training_history.csv belum tersedia.")
            else:
                st.dataframe(history_df, use_container_width=True)
                col_acc, col_loss = st.columns(2)
                with col_acc:
                    if {"accuracy", "val_accuracy"}.issubset(history_df.columns):
                        st.pyplot(plot_training(history_df, "accuracy", "val_accuracy", "Akurasi Model", "Accuracy"), use_container_width=True)
                with col_loss:
                    if {"loss", "val_loss"}.issubset(history_df.columns):
                        st.pyplot(plot_training(history_df, "loss", "val_loss", "Loss Model", "Loss"), use_container_width=True)

        with tab2:
            if report_df.empty:
                st.info("classification_report.csv belum tersedia.")
            else:
                st.dataframe(report_df, use_container_width=True)

        with tab3:
            if cm_df.empty:
                st.info("confusion_matrix.csv belum tersedia.")
            else:
                st.dataframe(cm_df, use_container_width=True)
                st.pyplot(plot_confusion_matrix(cm_df), use_container_width=True)

elif page == "📁 Dataset":
    st.subheader("Ringkasan Dataset")
    dataset_summary = load_dataset_summary()

    if dataset_summary.empty:
        st.warning("Ringkasan dataset belum tersedia. Jalankan training terlebih dahulu.")
        st.markdown("Struktur dataset yang direkomendasikan:")
        st.code(
            """dataset/
  Bacterial leaf blight/
    image_1.jpg
  Brown spot/
    image_2.jpg
  Leaf smut/
    image_3.jpg""",
            language="text",
        )
    else:
        total_images = int(dataset_summary["jumlah_gambar"].sum())
        st.metric("Total gambar", total_images)
        st.dataframe(dataset_summary, use_container_width=True, hide_index=True)
        st.bar_chart(dataset_summary.set_index("kelas")["jumlah_gambar"])

elif page == "📘 Panduan":
    st.subheader("Panduan Penggunaan")

    if class_names:
        st.markdown("### Solusi yang tersedia di aplikasi")
        solution_rows = get_all_solution_rows(class_names)
        st.dataframe(pd.DataFrame(solution_rows), use_container_width=True, hide_index=True)
        st.info(
            "Jika nama kelas dari dataset berbeda, aplikasi akan mencoba mencocokkan otomatis. "
            "Untuk menambah penyakit baru, edit file `disease_solutions.py`."
        )

    st.markdown("---")

    st.markdown("### A. Cara menjalankan paling mudah")
    render_commands()

    st.markdown("### B. Download dataset manual")
    st.write(
        "Buka halaman Kaggle dataset, download file ZIP, lalu ekstrak ke folder `dataset/`. "
        "Setelah itu jalankan `python train_model.py`."
    )

    st.markdown("### C. Training dengan model yang lebih bagus")
    st.write(
        "Default memakai `simple_cnn` karena tidak perlu internet. Jika ingin mencoba model yang biasanya lebih kuat, gunakan MobileNetV2."
    )
    st.code("python train_model.py --model mobilenet --epochs 5", language="bash")

    st.markdown("### D. Troubleshooting")
    st.markdown(
        """
- Jika muncul **Model belum dilatih**, jalankan `python train_model.py`.
- Jika muncul **dataset tidak valid**, pastikan folder `dataset/` berisi subfolder kelas.
- Jika Kaggle meminta API token, login ke Kaggle lalu buat API token dari Account Settings.
- Jika training lambat, kurangi epoch: `python train_model.py --epochs 5`.
        """
    )
