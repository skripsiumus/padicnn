"""
Keterangan dan solusi penanganan penyakit daun padi.

Catatan:
- Isi file ini menggunakan rekomendasi umum pengelolaan penyakit tanaman padi.
- Untuk penggunaan pestisida/fungisida/bakterisida, ikuti label produk, dosis resmi,
  dan anjuran penyuluh pertanian setempat.
"""

from __future__ import annotations

import re
from difflib import get_close_matches


DISEASE_SOLUTIONS = {
    "bacterial_leaf_blight": {
        "nama_tampilan": "Bacterial Leaf Blight / Hawar Daun Bakteri",
        "kategori": "Penyakit bakteri",
        "ringkasan": (
            "Hawar daun bakteri umumnya menyebabkan daun menguning, mengering, "
            "dan tampak seperti terbakar dari ujung atau tepi daun. Penyakit ini sering "
            "lebih berat pada kondisi lembap, hujan, dan genangan air."
        ),
        "gejala": [
            "Daun menguning sampai putih keabu-abuan, biasanya dari ujung atau pinggir daun.",
            "Pada serangan berat daun cepat kering dan tanaman tampak seperti terbakar.",
            "Dapat muncul cairan/ooze bakteri pada bagian daun muda yang terserang.",
        ],
        "penanganan": [
            "Pisahkan dan buang bagian tanaman yang terserang berat agar sumber infeksi berkurang.",
            "Perbaiki drainase; hindari air tergenang terlalu lama karena bakteri mudah menyebar melalui air.",
            "Kurangi pemupukan nitrogen berlebihan; gunakan pemupukan berimbang sesuai rekomendasi lahan.",
            "Gunakan benih sehat dan varietas yang lebih tahan pada musim tanam berikutnya.",
            "Jika serangan meluas, konsultasikan ke penyuluh pertanian untuk pilihan bakterisida yang legal dan sesuai lokasi.",
        ],
        "pencegahan": [
            "Gunakan benih bersertifikat/sehat.",
            "Bersihkan sisa tanaman sakit setelah panen.",
            "Atur jarak tanam agar sirkulasi udara lebih baik.",
            "Hindari melukai tanaman saat pemeliharaan karena luka dapat menjadi pintu masuk bakteri.",
        ],
    },
    "brown_spot": {
        "nama_tampilan": "Brown Spot / Bercak Cokelat",
        "kategori": "Penyakit jamur",
        "ringkasan": (
            "Bercak cokelat adalah penyakit jamur yang menimbulkan bercak cokelat pada daun. "
            "Penyakit ini sering berkaitan dengan tanaman lemah, kekurangan hara, atau kondisi lahan kurang optimal."
        ),
        "gejala": [
            "Bercak bulat atau oval berwarna cokelat pada daun.",
            "Bercak dapat memiliki bagian tengah abu-abu dan pinggir cokelat gelap.",
            "Pada serangan berat, daun mengering dan kemampuan fotosintesis menurun.",
        ],
        "penanganan": [
            "Buang daun yang terserang berat jika memungkinkan, terutama pada area pembibitan kecil.",
            "Perbaiki kesuburan tanah dengan pemupukan berimbang, terutama kalium dan unsur mikro bila diperlukan.",
            "Jaga sanitasi lahan dan bersihkan sisa tanaman sakit setelah panen.",
            "Gunakan benih sehat; perlakuan benih dapat dipertimbangkan sesuai rekomendasi setempat.",
            "Jika serangan berat, gunakan fungisida yang terdaftar untuk padi sesuai label dan arahan penyuluh.",
        ],
        "pencegahan": [
            "Gunakan varietas yang toleran atau tahan jika tersedia.",
            "Hindari kekurangan hara dan stres kekeringan.",
            "Lakukan rotasi atau pengelolaan lahan agar sumber inokulum jamur berkurang.",
            "Jangan menggunakan benih dari tanaman yang menunjukkan gejala berat.",
        ],
    },
    "leaf_smut": {
        "nama_tampilan": "Leaf Smut / Gosong Daun",
        "kategori": "Penyakit jamur",
        "ringkasan": (
            "Leaf smut disebabkan jamur dan biasanya tampak sebagai bintik kecil hitam pada daun. "
            "Pada banyak kasus penyakit ini tergolong ringan, tetapi tetap perlu dikendalikan agar tidak menjadi sumber spora."
        ),
        "gejala": [
            "Muncul bintik kecil hitam, agak menonjol, berbentuk garis/oval pada permukaan daun.",
            "Bintik dapat tersebar banyak tetapi biasanya tetap terpisah satu sama lain.",
            "Pada serangan berat, daun bisa menguning dan ujung daun mengering.",
        ],
        "penanganan": [
            "Kurangi sumber spora dengan membersihkan sisa daun/tanaman sakit setelah panen.",
            "Hindari pemupukan nitrogen berlebihan karena pertumbuhan daun terlalu rimbun dapat mendukung penyakit.",
            "Perbaiki jarak tanam dan sirkulasi udara agar kelembapan tidak terlalu tinggi.",
            "Gunakan benih sehat dan varietas yang lebih toleran jika tersedia.",
            "Bila infeksi berat dan meluas, konsultasikan penggunaan fungisida terdaftar kepada penyuluh pertanian.",
        ],
        "pencegahan": [
            "Lakukan sanitasi lahan dan buang sisa tanaman terinfeksi.",
            "Gunakan pemupukan berimbang, tidak berlebihan nitrogen.",
            "Pantau tanaman sejak fase vegetatif sampai generatif.",
            "Pastikan lahan tidak terlalu lembap dan tanaman tidak terlalu rapat.",
        ],
    },
    "blast": {
        "nama_tampilan": "Blast / Blas Padi",
        "kategori": "Penyakit jamur",
        "ringkasan": (
            "Blas padi dapat menyerang daun, leher malai, dan buku batang. Gejala pada daun sering berupa bercak lonjong "
            "atau belah ketupat dengan pusat abu-abu."
        ),
        "gejala": [
            "Bercak berbentuk lonjong/belah ketupat dengan tepi cokelat dan tengah abu-abu.",
            "Pada serangan berat, daun mengering dan bibit dapat mati.",
            "Jika menyerang leher malai, malai dapat hampa atau patah.",
        ],
        "penanganan": [
            "Gunakan varietas tahan/toleran blas pada musim tanam berikutnya.",
            "Hindari nitrogen berlebihan dan gunakan pemupukan berimbang.",
            "Jaga jarak tanam agar tidak terlalu rapat dan kelembapan tajuk berkurang.",
            "Bersihkan sisa tanaman sakit yang dapat menjadi sumber infeksi.",
            "Gunakan fungisida terdaftar bila diperlukan sesuai fase tanaman dan rekomendasi penyuluh.",
        ],
        "pencegahan": [
            "Gunakan benih sehat.",
            "Pantau gejala sejak persemaian.",
            "Hindari kondisi tanaman terlalu rimbun.",
            "Lakukan pengamatan rutin setelah hujan atau cuaca lembap.",
        ],
    },
    "healthy": {
        "nama_tampilan": "Healthy / Daun Sehat",
        "kategori": "Tidak terdeteksi penyakit utama",
        "ringkasan": (
            "Gambar terklasifikasi sebagai daun sehat. Tetap lakukan pemantauan berkala karena gejala awal penyakit "
            "kadang belum terlihat jelas pada foto."
        ),
        "gejala": [
            "Warna daun relatif hijau normal.",
            "Tidak tampak bercak dominan, gosong, atau bagian kering yang mencolok.",
        ],
        "penanganan": [
            "Lanjutkan pemupukan berimbang sesuai kebutuhan tanaman.",
            "Jaga pengairan dan drainase tetap baik.",
            "Lakukan monitoring hama dan penyakit secara rutin.",
            "Gunakan benih sehat dan sanitasi lahan pada musim tanam berikutnya.",
        ],
        "pencegahan": [
            "Pantau daun secara berkala, terutama setelah hujan atau kelembapan tinggi.",
            "Jaga jarak tanam dan kebersihan lahan.",
            "Hindari pemupukan nitrogen berlebihan.",
        ],
    },
}


ALIASES = {
    "bacterial_leaf_blight": [
        "bacterial leaf blight",
        "bacterial blight",
        "bacterial_leaf_blight",
        "bacterial_leaf_blight",
        "hawar daun bakteri",
        "blight",
    ],
    "brown_spot": [
        "brown spot",
        "brown_spot",
        "brownspot",
        "bercak cokelat",
        "bercak coklat",
    ],
    "leaf_smut": [
        "leaf smut",
        "leaf_smut",
        "leafsmut",
        "gosong daun",
        "smut",
    ],
    "blast": [
        "blast",
        "leaf blast",
        "rice blast",
        "blas",
        "blas padi",
    ],
    "healthy": [
        "healthy",
        "health",
        "normal",
        "sehat",
        "daun sehat",
    ],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_solution_key(class_name: str) -> str | None:
    normalized = normalize_text(class_name)

    for key, aliases in ALIASES.items():
        if normalized == normalize_text(key):
            return key
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if normalized == alias_norm or alias_norm in normalized or normalized in alias_norm:
                return key

    # Fallback fuzzy matching untuk variasi nama folder dataset
    choices = []
    key_by_choice = {}
    for key, aliases in ALIASES.items():
        for alias in aliases + [key]:
            alias_norm = normalize_text(alias)
            choices.append(alias_norm)
            key_by_choice[alias_norm] = key

    match = get_close_matches(normalized, choices, n=1, cutoff=0.72)
    if match:
        return key_by_choice[match[0]]

    return None


def get_solution_for_class(class_name: str) -> dict:
    key = detect_solution_key(class_name)
    if key and key in DISEASE_SOLUTIONS:
        data = DISEASE_SOLUTIONS[key].copy()
        data["matched"] = True
        data["key"] = key
        return data

    return {
        "matched": False,
        "key": "unknown",
        "nama_tampilan": class_name,
        "kategori": "Kelas belum ada di database solusi",
        "ringkasan": (
            "Nama kelas ini belum memiliki keterangan solusi khusus. Tambahkan data solusi baru pada file "
            "disease_solutions.py agar aplikasi dapat menampilkan rekomendasi yang lebih spesifik."
        ),
        "gejala": [
            "Periksa gejala visual secara langsung di lapangan.",
            "Bandingkan dengan beberapa sampel daun lain dari tanaman yang sama.",
        ],
        "penanganan": [
            "Ambil foto ulang dengan pencahayaan jelas dan fokus pada bagian daun yang sakit.",
            "Konsultasikan ke penyuluh pertanian atau ahli proteksi tanaman untuk diagnosis lapangan.",
            "Terapkan sanitasi lahan, pemupukan berimbang, dan pengairan yang baik sebagai langkah umum.",
        ],
        "pencegahan": [
            "Gunakan benih sehat dan varietas yang sesuai lokasi.",
            "Lakukan pemantauan rutin pada fase pertumbuhan awal sampai menjelang panen.",
        ],
    }


def get_all_solution_rows(class_names: list[str]) -> list[dict]:
    rows = []
    for name in class_names:
        solution = get_solution_for_class(name)
        rows.append(
            {
                "kelas_model": name,
                "nama_solusi": solution["nama_tampilan"],
                "kategori": solution["kategori"],
                "tersedia": "Ya" if solution["matched"] else "Belum khusus",
            }
        )
    return rows
