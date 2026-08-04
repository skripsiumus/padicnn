# Panduan Agar Model Hasil Google Colab Terbaca di Streamlit Cloud

Project ini sudah diperbaiki agar model hasil training Google Colab dapat terbaca langsung di Streamlit Cloud.

## Penyebab umum model tidak terbaca

Model di folder ini disimpan menggunakan Keras 3.13.2.  
Sebelumnya `requirements.txt` masih memakai:

```txt
tensorflow-cpu==2.15.0
keras==2.15.0
```

Versi tersebut dapat menyebabkan file `.keras` hasil Google Colab gagal dibaca.

## Perbaikan yang sudah dilakukan

1. `requirements.txt` diganti agar kompatibel dengan Keras 3:
   ```txt
   tensorflow==2.21.0
   keras==3.13.2
   ```

2. `runtime.txt` diganti menjadi:
   ```txt
   python-3.11
   ```

3. `config.py` dibuat otomatis mencari model dari beberapa lokasi:
   ```txt
   models/rice_disease_model.keras
   models/model_padi.keras
   model_padi.keras
   rice_disease_model.keras
   ```

4. `app.py` membaca model dengan:
   ```python
   tf.keras.models.load_model(str(MODEL_PATH), compile=False)
   ```

## File yang wajib ada untuk prediksi

Minimal file berikut harus ada di GitHub:

```txt
app.py
config.py
utils.py
disease_solutions.py
requirements.txt
runtime.txt
models/rice_disease_model.keras
models/class_names.txt
```

File tambahan yang juga sudah disediakan:

```txt
model_padi.keras
labels.txt
models/model_padi.keras
models/labels.txt
```

## Cara deploy ke Streamlit Cloud

1. Upload semua isi folder project ini ke GitHub.
2. Buka Streamlit Cloud.
3. Pilih repository GitHub.
4. Main file path: `app.py`.
5. Buka Advanced settings dan pilih Python 3.11 jika tersedia.
6. Deploy.
7. Jika sebelumnya sudah pernah error:
   - Manage app
   - Clear cache
   - Reboot app

## Setelah training ulang di Google Colab

Jika nanti Anda training ulang di Colab, download hasil:

```txt
models/rice_disease_model.keras
models/class_names.txt
```

Lalu upload/replace dua file tersebut ke GitHub.  
Streamlit Cloud akan otomatis membaca model baru setelah reboot.
