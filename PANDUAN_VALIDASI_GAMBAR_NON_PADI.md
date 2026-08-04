# Validasi Gambar Non-Padi

Versi ini menambahkan penyaring gambar di luar dataset sebelum hasil penyakit ditampilkan.

## Mekanisme

1. Memeriksa resolusi, pencahayaan, tekstur, serta proporsi warna hijau/kuning/cokelat yang lazim pada daun.
2. Menjalankan tiga variasi prediksi ringan (gambar asli, mirror, dan sedikit lebih terang).
3. Mengukur confidence, margin dua kelas teratas, entropy, dan konsistensi prediksi.
4. Jika gambar tidak memenuhi syarat, aplikasi menampilkan pesan:

> Maaf, gambar tersebut bukan foto penyakit daun padi. Sistem tidak dapat mengidentifikasi penyakit daun padi.

## Rekomendasi terbaik

Penyaring berbasis aturan tidak dapat menjamin 100% penolakan semua objek asing karena model awal hanya mengenal kelas penyakit padi. Solusi ilmiah yang lebih kuat adalah menambahkan kelas `Bukan Daun Padi` pada dataset, mengisinya dengan gambar negatif yang beragam, lalu melatih ulang model.

## Penyesuaian sensitivitas

Ambang batas tersedia di `config.py`. Jika foto daun padi yang benar terlalu sering ditolak, turunkan sedikit `RICE_MIN_CONFIDENCE` atau `RICE_MIN_LEAF_COLOR_RATIO`. Jika gambar asing masih sering lolos, naikkan nilai tersebut secara bertahap.
