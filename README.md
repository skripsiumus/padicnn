# Klasifikasi Penyakit Daun Padi Menggunakan CNN + Streamlit

Proyek ini adalah versi yang lebih mudah dan lebih rapi untuk membuat web klasifikasi penyakit daun padi dari dataset Kaggle:

`anshulm257/rice-disease-dataset`

Aplikasi dibuat dengan:

- **TensorFlow/Keras** untuk model CNN
- **Streamlit** untuk web interface
- **Matplotlib** untuk grafik hasil klasifikasi dan evaluasi
- **Scikit-learn** untuk classification report dan confusion matrix

---

## Isi Folder

```text
rice_disease_streamlit_cnn_easy/
├── app.py                    # Web Streamlit
├── train_model.py            # Training CNN
├── download_dataset.py       # Download dataset Kaggle
├── start_here.py             # Menu mudah untuk pemula
├── config.py                 # Konfigurasi path dan parameter
├── utils.py                  # Fungsi bantuan
├── requirements.txt          # Library yang dibutuhkan
├── README.md                 # Panduan
├── dataset/                  # Tempat dataset
├── models/                   # Tempat model hasil training
├── scripts/                  # Script cepat Windows/Linux
└── .streamlit/config.toml    # Tema tampilan Streamlit
```

---

## Cara Paling Mudah

Jalankan menu pemula:

```bash
python start_here.py
```

Lalu pilih menu:

1. Install library
2. Download dataset Kaggle
3. Training model CNN
4. Jalankan web Streamlit

---

## Cara Manual

### 1. Buat virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install library

```bash
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python download_dataset.py
```

Jika Kaggle meminta API token:

1. Login ke Kaggle.
2. Masuk ke **Account Settings**.
3. Pilih **Create New API Token**.
4. Simpan file `kaggle.json` sesuai instruksi Kaggle.

Atau download manual dari Kaggle, lalu ekstrak ke folder `dataset/`.

Struktur dataset yang benar:

```text
dataset/
  Nama_Kelas_1/
    gambar1.jpg
    gambar2.jpg
  Nama_Kelas_2/
    gambar3.jpg
    gambar4.jpg
```

Script training sudah bisa mencari folder kelas secara otomatis walaupun dataset punya subfolder tambahan.

### 4. Training model CNN

```bash
python train_model.py
```

Output akan muncul di folder `models/`:

```text
models/
├── rice_disease_model.keras
├── class_names.txt
├── training_history.csv
├── classification_report.csv
├── confusion_matrix.csv
├── dataset_summary.csv
├── accuracy_plot.png
├── loss_plot.png
└── model_info.json
```

### 5. Jalankan web

```bash
streamlit run app.py
```

---

## Versi Model

Default memakai CNN sederhana:

```bash
python train_model.py --model simple_cnn
```

Untuk hasil yang biasanya lebih bagus, coba transfer learning MobileNetV2:

```bash
python train_model.py --model mobilenet --epochs 15
```

Catatan: MobileNetV2 pertama kali membutuhkan internet untuk download bobot ImageNet.

---

## Fitur Web

Web Streamlit memiliki halaman:

1. **Beranda**  
   Menampilkan status model dan langkah penggunaan.

2. **Prediksi**  
   Upload gambar daun padi, lalu tampil:
   - hasil prediksi penyakit,
   - confidence,
   - tabel probabilitas tiap kelas,
   - grafik confidence.

3. **Grafik & Evaluasi**  
   Menampilkan:
   - grafik accuracy,
   - grafik loss,
   - classification report,
   - confusion matrix.

4. **Dataset**  
   Menampilkan jumlah gambar per kelas.

5. **Panduan**  
   Berisi perintah cepat dan troubleshooting.

---

## Perintah Cepat

Windows:

```bash
scripts\install.bat
scripts\download_dataset.bat
scripts\train.bat
scripts\run_app.bat
```

Mac/Linux:

```bash
bash scripts/install.sh
bash scripts/download_dataset.sh
bash scripts/train.sh
bash scripts/run_app.sh
```

---

## Troubleshooting

### Model belum tersedia

Jalankan:

```bash
python train_model.py
```

### Dataset tidak valid

Pastikan folder `dataset/` berisi minimal dua folder kelas yang masing-masing berisi gambar.

### Kaggle gagal download

Pastikan:

- Sudah install Kaggle CLI.
- Sudah login Kaggle.
- File `kaggle.json` sudah benar.

### Training terlalu lama

Coba kurangi epoch:

```bash
python train_model.py --epochs 5
```

Atau gunakan gambar lebih kecil:

```bash
python train_model.py --img-size 160 --epochs 10
```

---

## Catatan

Aplikasi ini dibuat untuk pembelajaran dan prototipe. Hasil prediksi tidak menggantikan pemeriksaan langsung oleh ahli pertanian.


## Fitur Solusi Penanganan Penyakit

Pada halaman **Prediksi**, setelah gambar diklasifikasikan, aplikasi akan menampilkan:

- nama penyakit hasil prediksi,
- ringkasan penyakit,
- gejala umum,
- cara menangani,
- langkah pencegahan.

File solusi berada di:

```text
disease_solutions.py
```

Untuk menambah penyakit baru, tambahkan data baru pada `DISEASE_SOLUTIONS` dan variasi nama kelas pada `ALIASES`.

> Catatan: rekomendasi penanganan bersifat umum. Untuk pestisida/fungisida/bakterisida, gunakan produk yang terdaftar, ikuti label, dan konsultasikan dengan penyuluh pertanian setempat.
