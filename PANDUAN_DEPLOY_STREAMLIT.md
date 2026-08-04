# Panduan Deploy ke Streamlit Cloud

## Struktur file yang wajib ada

```text
cnndaunpadi/
├── app.py
├── requirements.txt
├── runtime.txt
├── config.py
├── utils.py
├── disease_solutions.py
├── .streamlit/
│   └── config.toml
└── models/
    ├── rice_disease_model.keras      # wajib untuk prediksi
    ├── class_names.txt               # wajib untuk prediksi
    ├── training_history.csv          # opsional
    ├── confusion_matrix.csv          # opsional
    ├── classification_report.csv     # opsional
    ├── dataset_summary.csv           # opsional
    └── model_info.json               # opsional
```

## 1. Upload ke GitHub

Upload semua file ke repository GitHub. Pastikan `requirements.txt` dan `runtime.txt` berada sejajar dengan `app.py`.

## 2. Atur Streamlit Cloud

- Repository: pilih repository GitHub Anda
- Branch: `main`
- Main file path: `app.py`

## 3. Clear cache jika pernah error

Di Streamlit Cloud buka:

`Manage App → Settings → Clear cache → Reboot app`

## 4. Catatan model

Aplikasi tetap bisa terbuka walaupun model belum tersedia, tetapi menu prediksi belum bisa dipakai.
Untuk mengaktifkan prediksi, masukkan file berikut ke folder `models/`:

- `rice_disease_model.keras`
- `class_names.txt`

Jika model lebih dari 100 MB, gunakan Git LFS atau simpan model di tempat lain lalu unduh saat runtime.
