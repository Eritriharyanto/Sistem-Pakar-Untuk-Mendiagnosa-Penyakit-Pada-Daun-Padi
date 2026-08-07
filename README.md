# 🌾 Sistem Pakar untuk Mendiagnosa Penyakit pada Daun Padi

Proyek ini membangun sistem klasifikasi citra berbasis *deep learning* (transfer learning MobileNetV2) untuk mendeteksi penyakit pada daun padi, lalu memadukannya dengan basis pengetahuan sederhana (deskripsi & rekomendasi penanganan) sehingga keluaran sistem tidak hanya berupa label, melainkan juga penjelasan yang dapat digunakan oleh petani/pengguna — ciri khas dari sebuah **sistem pakar**.

## 📊 Dataset

Dataset yang digunakan adalah [Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification/data) dari Kaggle, yang berisi citra daun padi dengan beberapa kelas penyakit serta kelas daun sehat.

## 🔁 Alur Pengerjaan

1. **IMPORT** — Import library, cek versi TensorFlow, set seed untuk reproducibility, dan mount Google Drive (proyek dijalankan di Google Colab).
2. **EDA (Exploratory Data Analysis)** — Mengecek struktur folder & jumlah kelas, melihat distribusi jumlah gambar per kelas, dan menampilkan contoh gambar dari tiap kelas.
3. **TRAIN AND VAL** — Membuat data generator (augmentasi kuat untuk data training, tanpa augmentasi untuk data validasi) serta menghitung *class weight* untuk mengatasi ketidakseimbangan (*imbalance*) jumlah data antar kelas.
4. **MODEL** — Membangun model berbasis **MobileNetV2** (transfer learning + fine-tuning sebagian layer terakhir), meng-*compile* model, dan menyiapkan *callback* `EarlyStopping` & `ModelCheckpoint`.
5. **TRAINING** — Melatih model pada tahap awal (base model sebagian besar dibekukan), kemudian dilanjutkan dengan *fine-tuning* menggunakan *learning rate* yang lebih kecil.
6. **EVALUASI** — Menampilkan plot akurasi/loss selama training, *classification report*, *confusion matrix*, lalu menyimpan model akhir.
7. **PREDIKSI & REKOMENDASI (Modul Sistem Pakar)** — Basis pengetahuan berisi deskripsi & rekomendasi penanganan per jenis penyakit, fungsi prediksi + diagnosis otomatis, serta **Grad-CAM** untuk memvisualisasikan bagian citra yang menjadi dasar keputusan model.
8. **SUBMISSION** — Menjalankan prediksi ke seluruh data test dan menyimpan hasilnya ke `submission.csv`.

## 🧠 Arsitektur Model

- **Base model:** MobileNetV2 (pretrained `imagenet`, tanpa top layer)
- **Fine-tuning:** 30 layer terakhir dari base model dibuka agar ikut dilatih, sisanya dibekukan
- **Head tambahan:**
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.5)`
  - `Dense(NUM_CLASSES, activation='softmax')`
- **Optimizer:** Adam (`lr=1e-4` untuk training awal, `lr=1e-5` untuk fine-tuning)
- **Loss:** `categorical_crossentropy`

## 🖼️ Augmentasi Data

Augmentasi diterapkan secara agresif pada data training untuk mensimulasikan variasi kondisi foto dunia nyata (kamera HP, pencahayaan lapangan, sedikit blur), meliputi:
- Rotasi, zoom, horizontal & vertical flip
- Shear, width/height shift
- Brightness & channel shift
- Blur ringan (Gaussian blur) dan jitter kontras acak (custom preprocessing)

Data validasi **tidak** diaugmentasi agar tetap merepresentasikan kondisi data asli.

## ⚖️ Penanganan Data Imbalance

Karena jumlah gambar antar kelas penyakit tidak seimbang, digunakan `class_weight='balanced'` dari scikit-learn saat training agar model tidak bias terhadap kelas mayoritas.

## 📈 Evaluasi

Evaluasi dilakukan menggunakan:
- Plot akurasi & loss (train vs validation)
- **Classification report** (precision, recall, f1-score per kelas)
- **Confusion matrix** — penting untuk dataset imbalance, karena akurasi keseluruhan yang tinggi bisa menyembunyikan kesalahan prediksi pada kelas minoritas (mis. `bacterial_panicle_blight`).

## 🩺 Modul Sistem Pakar

Setelah model CNN memprediksi label penyakit, sistem memetakan hasil tersebut ke sebuah basis pengetahuan (`DISEASE_INFO`) yang berisi:
- **Deskripsi** singkat mengenai penyakit tersebut
- **Rekomendasi** penanganan yang bisa dilakukan

Fungsi utama:
- `predict_image(img_path)` — mengembalikan label prediksi & confidence
- `diagnose(img_path)` — mencetak diagnosis lengkap (label, confidence, deskripsi, rekomendasi)

### Grad-CAM

Untuk menjawab pertanyaan "kenapa model mendiagnosis begini?", disediakan visualisasi **Grad-CAM** yang menyorot area citra daun yang paling memengaruhi keputusan model — membuat sistem lebih dapat dijelaskan (*explainable*), bukan sekadar memberi label.

## 📦 Output yang Dihasilkan

| File | Keterangan |
|---|---|
| `model_padi_best.keras` | Model terbaik hasil `ModelCheckpoint` |
| `model_padi_final.keras` | Model akhir setelah training + fine-tuning |
| `training_history.json` | Riwayat akurasi & loss selama training |
| `labels.json` | Mapping indeks kelas ke nama kelas |
| `submission.csv` | Hasil prediksi seluruh data test (format Kaggle: `image_id`, `label`) |
| `submission_with_confidence.csv` | Versi lengkap hasil prediksi beserta nilai confidence |

## 🛠️ Library yang Digunakan

- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- OpenCV (`opencv-python-headless`)
- scikit-learn (`class_weight`, `classification_report`, `confusion_matrix`)

## ▶️ Cara Menjalankan

1. Buka notebook `my-padi.ipynb` di **Google Colab**.
2. Unduh dataset dari [Kaggle](https://www.kaggle.com/competitions/paddy-disease-classification/data) dan letakkan di Google Drive dengan struktur folder:
   ```
   /content/drive/MyDrive/padi-saya/
   ├── train_images/
   │   ├── <nama_kelas_1>/
   │   ├── <nama_kelas_2>/
   │   └── ...
   └── test_images/
   ```
3. Jalankan seluruh cell secara berurutan mulai dari bagian **IMPORT** hingga **SUBMISSION**.
4. Model, riwayat training, dan hasil prediksi akan tersimpan otomatis sesuai tabel *output* di atas.

## 📌 Catatan

- Notebook ini dirancang untuk dijalankan di **Google Colab** (menggunakan `google.colab.drive` untuk mount Google Drive).
- Proses training terdiri dari dua tahap: training awal (sebagian besar base model dibekukan) lalu fine-tuning (30 layer terakhir MobileNetV2 dibuka dengan learning rate lebih kecil).
- Augmentasi data pada versi terbaru (v2) telah diperkuat dan dipisahkan dari generator validasi — jika melakukan perubahan pada bagian augmentasi, training perlu dijalankan ulang dari awal.
