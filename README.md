# FURAP-AI: Smart Tourism Destination Recommendation and Itinerary Planner

## AI for Smart Tourism Experience

FURAP-AI adalah capstone project bertema **AI for Smart Tourism Experience** yang dikembangkan untuk membantu pengguna mendapatkan rekomendasi destinasi wisata dan menyusun rencana perjalanan secara lebih praktis. Sistem ini menggabungkan pendekatan **Machine Learning** untuk rekomendasi destinasi dan itinerary budgeting, serta **Deep Learning/CNN** untuk mengenali destinasi wisata melalui gambar.

---

## Tentang Proyek

Perencanaan wisata sering membutuhkan waktu karena pengguna perlu mencari destinasi, menyesuaikan budget, menentukan durasi perjalanan, memperkirakan biaya, serta mencari informasi lokasi secara manual. FURAP-AI hadir sebagai sistem rekomendasi wisata yang membantu pengguna memilih destinasi berdasarkan preferensi perjalanan.

Pada proyek ini, sistem dibagi menjadi dua modul utama:

1. **Machine Learning Module**  
   Digunakan untuk fitur itinerary budgeting dan rekomendasi destinasi berdasarkan preferensi pengguna seperti kategori wisata, budget, durasi perjalanan, jumlah orang, transportasi, penginapan, estimasi jarak, dan total biaya.

2. **Deep Learning Module**  
   Digunakan untuk fitur upload image atau take photo agar sistem dapat mengenali destinasi wisata dari gambar menggunakan pendekatan Convolutional Neural Network (CNN).

---

## Fitur Utama

### 1. Itinerary Budgeting & Recommendation

Fitur ini membantu pengguna mendapatkan rekomendasi destinasi wisata dan estimasi rencana perjalanan berdasarkan input seperti:

- kategori wisata
- budget
- durasi perjalanan
- jumlah orang
- jenis transportasi
- preferensi penginapan
- estimasi jarak
- estimasi total biaya

Model Machine Learning digunakan untuk menentukan apakah sebuah destinasi layak direkomendasikan atau tidak berdasarkan kombinasi preferensi pengguna dan karakteristik destinasi.

### 2. Tourist Landmark Recognition

Fitur ini memungkinkan pengguna untuk mengunggah gambar atau mengambil foto destinasi wisata. Model CNN akan menganalisis gambar tersebut dan memprediksi nama destinasi wisata yang sesuai.

Output dari fitur ini meliputi:

- nama destinasi wisata
- deskripsi singkat
- kategori wisata
- lokasi destinasi
- rekomendasi destinasi serupa

### 3. Google Maps API Integration

Google Maps API direncanakan digunakan pada tahap aplikasi Streamlit untuk mendukung:

- informasi lokasi
- rute perjalanan
- estimasi jarak
- estimasi waktu tempuh
- pendukung perhitungan biaya transportasi

Google Maps API tidak digunakan pada proses training model, tetapi digunakan pada tahap aplikasi setelah model menghasilkan rekomendasi.

---

## Metode Pengembangan

Proyek ini mengikuti pendekatan **CRISP-DM**:

1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  
6. Deployment Preparation  

---

## Teknologi yang Digunakan

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity
- KNN
- Joblib

### Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)
- Image Augmentation
- Pillow

### Data Visualization

- Plotly
- Matplotlib

### Web Application

- Streamlit

### Deployment & Monitoring

- Docker
- Grafana

### External API

- Google Maps API
