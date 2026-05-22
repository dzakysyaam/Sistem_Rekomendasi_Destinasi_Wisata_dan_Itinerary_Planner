# FURAP-AI Docker Ready

FURAP-AI adalah sistem rekomendasi wisata berbasis AI dengan dua modul utama:

1. **Tempat Wisata**  
   CNN digunakan untuk membaca gambar destinasi dari upload/take photo. Hasil prediksi gambar digunakan untuk mengambil detail destinasi dan rekomendasi wisata serupa.

2. **Itinerary Budgeting**  
   TF-IDF + KNN digunakan untuk mencocokkan preferensi pengguna dengan dataset destinasi. Sistem juga menampilkan estimasi budget dasar untuk perjalanan.

## Struktur Folder

```text
FURAP_AI_Docker_Ready/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .streamlit/
│   └── config.toml
└── models/
    ├── furap_AI_model.pkl
    ├── furap_cnn_model.keras
    ├── furap_content_knn.pkl
    ├── furap_destination_base.pkl
    ├── furap_destination_tfidf.pkl
    └── furap_cnn_classes.txt
```

## Cara Running dengan Docker

Masuk ke folder project:

```powershell
cd C:\Users\Syam\pijak\Furap-AI
```

Build image:

```powershell
docker build -t furap-ai .
```

Jalankan container:

```powershell
docker run --rm -p 8501:8501 furap-ai
```

Buka browser:

```text
http://localhost:8501
```

Kalau port 8501 sudah dipakai:

```powershell
docker run --rm -p 8502:8501 furap-ai
```

Lalu buka:

```text
http://localhost:8502
```
