from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
import requests
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from math import radians, sin, cos, sqrt, atan2


st.set_page_config(
    page_title="FURAP Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
css_file_path = BASE_DIR / "style.css"

if css_file_path.exists():
    st.markdown(f"<style>{css_file_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.error("⚠️ Berkas 'style.css' tidak ditemukan! Pastikan file style.css berada di folder yang sama dengan app.py agar tema premium aktif.")
    

# 2. Monitoring System menggunakan Promotheus
@st.cache_resource
def init_monitoring():
    try:
        start_http_server(8000)
    except OSError:
        pass
    return {
        "image_prediction_total": Counter(
            "furap_image_prediction_total",
            "Total image prediction requests",
            ["predicted_class"]
        ),
        "image_prediction_latency": Histogram(
            "furap_image_prediction_latency_seconds",
            "Image prediction latency in seconds"
        ),
        "image_prediction_confidence": Gauge(
            "furap_image_prediction_confidence",
            "Last image prediction confidence",
            ["predicted_class"]
        ),
        "itinerary_request_total": Counter(
            "furap_itinerary_request_total",
            "Total itinerary generation requests"
        ),
        "model_loaded_status": Gauge(
            "furap_model_loaded",
            "Model loading status. 1 means loaded, 0 means failed"
        ),
    }

metrics = init_monitoring()

# 3. Session State Engine bawaan Streamlit
if 'selected_destination' not in st.session_state:
    st.session_state.selected_destination = None
if 'selected_destination_detail' not in st.session_state:
    st.session_state.selected_destination_detail = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'location_loaded' not in st.session_state:
    st.session_state.location_loaded = False
if 'last_selected' not in st.session_state:
    st.session_state.last_selected = None
if 'camera_enabled' not in st.session_state:
    st.session_state.camera_enabled = False
if 'last_analysis_result' not in st.session_state:
    st.session_state.last_analysis_result = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📍 Tempat Wisata"

# 3. Fungsi Helper dan Utilitas
def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def pretty_name(text: str) -> str:
    return str(text).replace("_", " ").title()

def rupiah(value: float | int) -> str:
    try:
        return f"Rp{int(value):,}".replace(",", ".")
    except Exception:
        return "Rp0"

def maps_link(row: pd.Series) -> str:
    lat = row.get("latitude", None)
    lon = row.get("longitude", None)
    name = row.get("nama_wisata", "Destinasi Wisata")

    if pd.notna(lat) and pd.notna(lon):
        return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

    query = str(name).replace(" ", "+")
    return f"https://www.google.com/maps/search/?api=1&query={query}"

def safe_text(value, default="-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_user_location() -> dict:
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'lat': data.get('latitude', -6.2088),
                'lon': data.get('longitude', 106.8456),
                'city': data.get('city', 'Jakarta'),
                'country': data.get('country_name', 'Indonesia')
            }
    except Exception:
        pass
    
    return {
        'lat': -6.2088,
        'lon': 106.8456,
        'city': 'Jakarta',
        'country': 'Indonesia'
    }

# 4. Model Assets Loading Engine
@st.cache_resource
def load_assets():
    assets = {}

    cnn_path = MODEL_DIR / "furap_cnn_model.keras"
    tfidf_path = MODEL_DIR / "furap_destination_tfidf.pkl"
    base_path = MODEL_DIR / "furap_destination_base.pkl"
    knn_path = MODEL_DIR / "furap_content_knn.pkl"
    ml_path = MODEL_DIR / "furap_AI_model.pkl"
    class_path = MODEL_DIR / "furap_cnn_classes.txt"

    required_files = [cnn_path, tfidf_path, base_path, knn_path, class_path]
    missing = [p.name for p in required_files if not p.exists()]

    if missing:
        raise FileNotFoundError(f"File belum lengkap di folder models/: {missing}")

    assets["cnn_model"] = tf.keras.models.load_model(cnn_path, compile=False)
    assets["tfidf"] = joblib.load(tfidf_path)
    assets["destination_base"] = joblib.load(base_path)
    assets["content_knn"] = joblib.load(knn_path)
    assets["class_names"] = [
        line.strip()
        for line in class_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    try:
        assets["ml_model"] = joblib.load(ml_path) if ml_path.exists() else None
        assets["ml_model_status"] = "Siap digunakan" if ml_path.exists() else "Belum tersedia"
    except Exception:
        assets["ml_model"] = None
        assets["ml_model_status"] = "Belum dapat dimuat"

    return assets

try:
    assets = load_assets()
    metrics["model_loaded_status"].set(1)
except Exception as error:
    metrics["model_loaded_status"].set(0)
    st.error("Beberapa file sistem belum berhasil dimuat.")
    st.code(str(error))
    st.stop()

cnn_model = assets["cnn_model"]
tfidf = assets["tfidf"]
destination_base = assets["destination_base"].copy()
content_knn = assets["content_knn"]
class_names = assets["class_names"]
ml_model_status = assets["ml_model_status"]

if "class_name" not in destination_base.columns:
    destination_base["class_name"] = destination_base["nama_wisata"].apply(slugify)

if "teks_destinasi" not in destination_base.columns:
    text_columns = [
        col for col in ["nama_wisata", "kategori_wisata", "kota_tujuan", "alamat", "deskripsi_wisata"]
        if col in destination_base.columns
    ]
    destination_base["teks_destinasi"] = (
        destination_base[text_columns]
        .astype(str)
        .agg(" ".join, axis=1)
    )


# 5. Core AI dan Logika Bisnis
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_place(image: Image.Image, top_n: int = 5):
    start_time = time.time()
    processed = preprocess_image(image)
    prediction = cnn_model.predict(processed, verbose=0)[0]
    latency = time.time() - start_time
    metrics["image_prediction_latency"].observe(latency)
    top_indices = prediction.argsort()[-top_n:][::-1]
    top_predictions = []
    for idx in top_indices:
        class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        top_predictions.append({
            "class_name": class_name,
            "confidence": float(prediction[idx]),
        })
    best = top_predictions[0]
    metrics["image_prediction_total"].labels(
        predicted_class=best["class_name"]
    ).inc()
    metrics["image_prediction_confidence"].labels(
        predicted_class=best["class_name"]
    ).set(best["confidence"])
    return best["class_name"], best["confidence"], top_predictions

def recommend_by_query(query: str, top_n: int = 5, exclude_class: str | None = None, user_location: dict = None) -> pd.DataFrame:
    query_vec = tfidf.transform([query])
    all_neighbors = min(top_n + 5, len(destination_base))
    distances, indices = content_knn.kneighbors(query_vec, n_neighbors=all_neighbors)

    result = destination_base.iloc[indices[0]].copy()
    result["similarity_score"] = 1 - distances[0]

    if exclude_class:
        result = result[result["class_name"] != exclude_class]

    if user_location and "latitude" in result.columns and "longitude" in result.columns:
        distances_list = []
        for idx, row in result.iterrows():
            if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                distance = calculate_distance(
                    user_location['lat'], 
                    user_location['lon'],
                    row['latitude'], 
                    row['longitude']
                )
                distances_list.append(distance)
            else:
                distances_list.append(None)
        result["jarak_km"] = distances_list
        result = result.sort_values(["jarak_km", "similarity_score"], ascending=[True, False])
    else:
        result["jarak_km"] = None

    return result.head(top_n).reset_index(drop=True)

def get_destination_detail(class_name: str, user_location: dict = None) -> pd.DataFrame:
    matched = destination_base[destination_base["class_name"] == class_name]
    if not matched.empty:
        result = matched.head(1).copy()
        if user_location and "latitude" in result.columns and "longitude" in result.columns:
            row = result.iloc[0]
            if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                distance = calculate_distance(
                    user_location['lat'],
                    user_location['lon'],
                    row['latitude'],
                    row['longitude']
                )
                result["jarak_km"] = distance
            else:
                result["jarak_km"] = None
        return result

    query = pretty_name(class_name)
    return recommend_by_query(query, top_n=1, user_location=user_location)

def estimate_budget_with_distance(days: int, people: int, transport: str, lodging: str, budget: int, distance_km: float = 0) -> dict:
    meal_cost = days * people * 75000
    ticket_cost = days * people * 30000

    if transport == "Motor":
        transport_rate_per_km = 2000
    elif transport == "Mobil":
        transport_rate_per_km = 4500
    else:
        transport_rate_per_km = 3000
    
    transport_cost = (distance_km * transport_rate_per_km * 2) * days

    lodging_rate = {
        "Tanpa Penginapan": 0,
        "Hemat": 180000,
        "Standar": 350000,
        "Nyaman": 550000,
    }.get(lodging, 0)

    lodging_cost = lodging_rate * max(days - 1, 0)
    total = meal_cost + ticket_cost + transport_cost + lodging_cost

    return {
        "makan": meal_cost,
        "tiket": ticket_cost,
        "transportasi": transport_cost,
        "penginapan": lodging_cost,
        "total": total,
        "status": "Masih sesuai budget" if total <= budget else "Melebihi budget",
        "jarak_tempuh": distance_km * 2 * days
    }


# 6. Global Layout & Navigation Banner
st.markdown(
    """
<div class="hero">
    <div class="hero-content">
        <div class="hero-label">FURAP Travel Planner</div>
        <div class="hero-title">Rencanakan wisata dengan lebih praktis.</div>
        <div class="hero-subtitle">
            Temukan destinasi, susun estimasi perjalanan, dan kenali tempat wisata dari gambar.
            Dirancang agar rekomendasi mudah dibaca, nyaman digunakan, dan siap dipresentasikan.
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

menu_options = ["📍 Tempat Wisata", "🗓️ Rencana Perjalanan", "👥 Tentang Aplikasi"]
selected_menu = st.radio(
    "Navigasi Menu",
    menu_options,
    index=menu_options.index(st.session_state.active_tab),
    horizontal=True,
    label_visibility="collapsed"
)

# 7. Sidebar Control di samping kanan
with st.sidebar:
    st.markdown("🗺️ Lokasi Anda")
    
    if st.button("📍 Dapatkan Lokasi Saya", use_container_width=True):
        with st.spinner("Mendapatkan lokasi..."):
            st.session_state.user_location = get_user_location()
            st.session_state.location_loaded = True
            st.success(f"📍 Lokasi: {st.session_state.user_location['city']}")
            st.rerun()
    
    if st.session_state.location_loaded and st.session_state.user_location:
        loc = st.session_state.user_location
        st.markdown(
            f"""
            <div class="location-card">
                <b>📍 Lokasi Terdeteksi</b><br>
                Kota: {loc['city']}<br>
                Koordinat: {loc['lat']:.4f}, {loc['lon']:.4f}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("🔄 Refresh Lokasi", use_container_width=True):
            st.session_state.user_location = None
            st.session_state.location_loaded = False
            st.rerun()
    else:
        st.info("💡 Klik tombol di atas untuk sinkronisasi GPS lokasi Anda.")

    st.markdown("---")
    st.markdown("🎯 Destinasi Tujuan")
    
    if st.session_state.selected_destination_detail is not None:
        dest = st.session_state.selected_destination_detail
        st.markdown(
            f"""
            <div class="destination-badge">
                <b>🎯 {dest.get('nama', '-')}</b><br>
                Jarak: {dest.get('jarak', 0):.1f} km<br>
                Wilayah: {dest.get('kota', '-')}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("🗑️ Hapus Pilihan", use_container_width=True):
            st.session_state.selected_destination = None
            st.session_state.selected_destination_detail = None
            st.session_state.active_tab = "📍 Tempat Wisata"
            st.rerun()
    else:
        st.info("🏳️ Belum ada destinasi tujuan yang dikunci.")
    
    with st.expander("🛠️ Info Sistem Monitoring"):
        st.write("Status: Active Running")
        st.write("Artifacts Info:", {
            'selected_destination': st.session_state.selected_destination,
            'selected_destination_detail': st.session_state.selected_destination_detail is not None,
            'location_loaded': st.session_state.location_loaded
        })


# 8. Fitur Upload dan Analisis Gambar Destinasi
if selected_menu == "📍 Tempat Wisata":
    st.markdown('<div class="section-title">Kenali destinasi dari gambar</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("📸 Unggah gambar destinasi")

        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        camera_file = None
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            if not st.session_state.camera_enabled:
                if st.button("📷 Aktifkan Kamera", key="enable_camera_btn", use_container_width=True):
                    st.session_state.camera_enabled = True
                    st.rerun()
            else:
                st.markdown('<div class="upload-pop">✓ Kamera sistem siap.</div>', unsafe_allow_html=True)
        with cam_col2:
            if st.session_state.camera_enabled:
                if st.button("❌ Matikan Kamera", key="disable_camera_btn", use_container_width=True):
                    st.session_state.camera_enabled = False
                    st.rerun()

        if st.session_state.camera_enabled:
            camera_file = st.camera_input(
                "Ambil foto langsung",
                label_visibility="collapsed"
            )

        image_file = uploaded_file or camera_file

        if "last_uploaded_name" not in st.session_state:
            st.session_state.last_uploaded_name = None

        if image_file is not None:
            current_name = getattr(image_file, "name", "camera_input")
            if st.session_state.last_uploaded_name != current_name:
                st.session_state.last_uploaded_name = current_name
                st.session_state.last_analysis_result = None
                
                # Efek bounce untuk Gambar berhasil diunggah 
                st.markdown('<div class="bounce-toast">📸 Gambar berhasil diunggah!</div>', unsafe_allow_html=True)

            image = Image.open(image_file)
            st.image(image, caption="Pratinjau Foto", use_container_width=True)
        else:
            image = None

        # Tombol Analisis Gambar standar
        run_image = st.button("Analisis Gambar", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True) 

        st.markdown(
            """
<div class="card-compact">
    💡 <b>Tips gambar yang bagus</b>
    <p class="muted">
        Untuk akurasi maksimal, pilih foto yang tidak blur, cukup terang, dan memperlihatkan landmark utama 
        (contoh: gerbang masuk utama, prasasti nama tempat, atau arsitektur ikonik bangunan pariwisata Bandung).
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        hasil_container = st.container()

        if run_image:
            if image is None:
                with hasil_container:
                    st.markdown(
                        '<div class="warning-pop">⚠️ Silakan unggah gambar atau aktifkan kamera terlebih dahulu.</div>',
                        unsafe_allow_html=True
                    )
            else:
                with st.spinner("Gambar sedang di analisis..."):
                    predicted_class, confidence, top_predictions = predict_place(image)
                    user_loc = st.session_state.user_location if st.session_state.location_loaded else None
                    detail = get_destination_detail(predicted_class, user_location=user_loc)

                # Efek bounce untuk container Analisis Selesai
                st.markdown('<div class="bounce-toast-success">Analisis selesai!</div>', unsafe_allow_html=True)

                if not detail.empty:
                    row = detail.iloc[0]
                    nama_wisata = safe_text(row.get("nama_wisata", pretty_name(predicted_class))).title()
                    kategori = safe_text(row.get("kategori_wisata", "-")).title()
                    deskripsi = safe_text(row.get("deskripsi_wisata", "Deskripsi lengkap belum tersedia."))
                    alamat = safe_text(row.get("alamat", "-"))
                    jarak = row.get("jarak_km", None)
                    kota = safe_text(row.get("kota_tujuan", "-"))
                    lat = row.get("latitude", None)
                    lon = row.get("longitude", None)
                    
                    st.session_state.last_analysis_result = {
                        "has_detail": True,
                        "predicted_class": predicted_class,
                        "confidence": float(confidence),
                        "top_predictions": top_predictions,
                        "row": row.to_dict(),
                        "nama_wisata": nama_wisata,
                        "kategori": kategori,
                        "deskripsi": deskripsi,
                        "alamat": alamat,
                        "jarak": float(jarak) if (jarak is not None and pd.notna(jarak)) else 0,
                        "jarak_tersedia": bool(jarak is not None and pd.notna(jarak)),
                        "kota": str(kota),
                        "lat": float(lat) if (lat is not None and pd.notna(lat)) else None,
                        "lon": float(lon) if (lon is not None and pd.notna(lon)) else None,
                    }
                else:
                    st.session_state.last_analysis_result = {
                        "has_detail": False,
                        "predicted_class": predicted_class,
                        "confidence": float(confidence),
                        "top_predictions": top_predictions,
                    }

        analysis = st.session_state.last_analysis_result

        with hasil_container:
            if analysis:
                predicted_class = analysis.get("predicted_class")
                confidence = float(analysis.get("confidence", 0))
                top_predictions = analysis.get("top_predictions", [])

                if analysis.get("has_detail"):
                    row = pd.Series(analysis.get("row", {}))
                    nama_wisata = analysis.get("nama_wisata", pretty_name(predicted_class))
                    kategori = analysis.get("kategori", "-")
                    deskripsi = analysis.get("deskripsi", "")
                    alamat = analysis.get("alamat", "-")
                    jarak = analysis.get("jarak", 0)
                    jarak_tersedia = analysis.get("jarak_tersedia", False)
                    lat = analysis.get("lat")
                    lon = analysis.get("lon")

                    st.markdown(
                        f"""
    <div class="result-card">
        <div class="result-title">🎯 {nama_wisata}</div>
        <div class="badge">📂 Kategori: {kategori}</div>
    """,
                        unsafe_allow_html=True,
                    )

                    if jarak_tersedia:
                        st.markdown(f'<div class="badge">🚗 Jarak: {jarak:.1f} km dari posisi Anda</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="badge-soft">🌐 Jarak dari koordinat Anda tidak dihitung (GPS Nonaktif)</div>', unsafe_allow_html=True)

                    st.markdown(
                        f"""
        <p style="margin-top: 12px; font-size: 14.5px; line-height:1.6; color:#334155;">{deskripsi}</p>
        <p class="muted" style="margin-top:10px;">📍 <b>Alamat Resmi:</b> {alamat}</p>
    </div>
    """,
                        unsafe_allow_html=True,
                    )

                    if confidence < 0.60:
                        st.markdown(
                            '<div class="warning-pop">⚠️ Gambar kurang akurat. Gunakan foto/gambar dengan pencahayaan yang lebih baik.</div>',
                            unsafe_allow_html=True
                        )

                    maps_url = maps_link(row)
                    st.link_button("🗺️ Navigasi Google Maps", maps_url, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📌 Kunci Sebagai Tujuan", key="select_destination_btn", use_container_width=True):
                            st.session_state.selected_destination = predicted_class
                            st.session_state.selected_destination_detail = {
                                'nama': nama_wisata,
                                'jarak': float(jarak) if jarak_tersedia else 0,
                                'jarak_tersedia': bool(jarak_tersedia),
                                'kota': analysis.get("kota", "-"),
                                'lat': lat,
                                'lon': lon
                            }
                            st.session_state.active_tab = "🗓️ Rencana Perjalanan"
                            st.success(f"🎯 {nama_wisata} dipilih, atur itinerary!")
                            time.sleep(0.5)
                            st.rerun()

                    with col2:
                        if st.session_state.selected_destination == predicted_class:
                            st.markdown('<div class="upload-pop" style="text-align:center; margin:0; padding:10px;">✓ Destinasi Terkunci</div>', unsafe_allow_html=True)

                    if lat is not None and lon is not None:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.map(
                            pd.DataFrame([{"lat": lat, "lon": lon}]),
                            zoom=12,
                            use_container_width=True
                        )

                    # Kolom setelah kolom maps
                    st.markdown("Destinasi Lain yang cocok")
                    st.caption("Daftar tempat wisata alternatif berdasarkan kemiripan bentuk foto Anda.")
                    top_df = pd.DataFrame([
                        {
                            "Destinasi Wisata": pretty_name(item["class_name"]),
                            "Tingkat Kemiripan Foto": f"{item['confidence'] * 100:.2f}%"
                        }
                        for item in top_predictions
                    ])
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("Rekomendasi Destinasi Lainnya")
                    user_loc = st.session_state.user_location if st.session_state.location_loaded else None
                    recs = recommend_by_query(
                        row.get("teks_destinasi", predicted_class),
                        top_n=4,
                        exclude_class=row.get("class_name"),
                        user_location=user_loc
                    )

                    if recs.empty:
                        st.info("Tidak ada klaster destinasi pariwisata sejenis yang ditemukan.")
                    else:
                        for _, rec in recs.iterrows():
                            jarak_text = ""
                            if rec.get("jarak_km") is not None and pd.notna(rec.get("jarak_km")):
                                jarak_text = f" · 🗺️ {rec['jarak_km']:.1f} km"

                            st.markdown(
                                f"""
    <div class="card-compact">
        <b style="color:#102A43; font-size:14.5px;">{safe_text(rec.get('nama_wisata', '-')).title()}</b>
        <div class="muted" style="font-size:12px; margin: 2px 0 6px 0;">📂 {safe_text(rec.get('kategori_wisata', '-')).title()} · 💎 Kepadatan Matriks Kesamaan {float(rec.get('similarity_score', 0)) * 100:.1f}%{jarak_text}</div>
        <p style="font-size:13px; color:#475569; line-height:1.5;">{safe_text(rec.get('deskripsi_wisata', ''))}</p>
    </div>
    """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.markdown(
                        f"""
    <div class="result-card">
        <div class="result-title">🎯 {pretty_name(predicted_class)}</div>
        <div class="badge-soft">📈 Confidence: {confidence * 100:.2f}%</div>
        <p class="muted">Metadata detail destinasi pariwisata ini belum diekstraksi ke basis data lokal.</p>
    </div>
    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
    <div class="card" style="text-align:center; padding: 40px 20px;">
        <span style="font-size:40px;">🖼️</span>
        <h4 style="margin-top:10px; color:#102A43;">Hasil akan ditampilkan disini</h4>
        <p class="muted" style="max-width:460px; margin: 0 auto;">
            Setelah gambar diunggah dan dianalisis, halaman ini akan menampilkan nama destinasi, tingkat keyakinan, deskripsi, lokasi, dan rekomendasi tempat serupa
        </p>
    </div>
    """,
                    unsafe_allow_html=True,
                )

                if {"latitude", "longitude"}.issubset(destination_base.columns):
                    map_df = (
                        destination_base[["latitude", "longitude"]]
                        .dropna()
                        .rename(columns={"latitude": "lat", "longitude": "lon"})
                    )
                    if not map_df.empty:
                        st.map(map_df, zoom=9, use_container_width=True)



## Halaman Rencana Perjalanan

elif selected_menu == "🗓️ Rencana Perjalanan":
    st.markdown('<div class="section-title">Susun rencana perjalanan</div>', unsafe_allow_html=True)
    
    if st.session_state.selected_destination_detail is None:
        st.markdown(
            """
            <div class="card" style="text-align: center; padding: 60px 20px; background:#FFFFFF; border:1px solid #E2E8F0;">
                <span style="font-size: 50px;">📍</span>
                <h3 style="color: #102A43; margin-top: 15px; font-weight:800;">Destinasi Belum Terkunci</h3>
                <p class="muted" style="max-width: 520px; margin: 4px auto 20px auto; font-size:14px;">
                    Silakan buka tab menu <b>"📍 Tempat Wisata"</b> terlebih dahulu, unggah foto landmark wisata Bandung pilihan Anda, lalu tekan tombol <b>"Kunci Sebagai Tujuan"</b> untuk mengaktifkan sistem kalkulator itinerary.
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        dest = st.session_state.selected_destination_detail
        if dest.get('jarak_tersedia', False):
            jarak_badge = f"<div class=\"badge\">🚗 Jarak: {dest.get('jarak', 0):.1f} km dari titik GPS Anda</div>"
        else:
            jarak_badge = '<div class="badge-soft">🌐 GPS Off · Nyalakan GPS untuk akurasi titik koordinat</div>'

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title" style="font-size: 22px; margin-bottom: 4px;">✈️ Simulasi Itinerary: Perjalanan ke {dest.get('nama', '-')}</div>
                {jarak_badge} · <span class="muted">Wilayah: {dest.get('kota', '-')}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        left, right = st.columns([1, 1.1])

        with left:
            st.markdown("Silahkan pilih ketentuan untuk simulasi Itinerary!")
            
            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                days = st.number_input("Durasi Wisata (Hari)", min_value=1, max_value=5, value=2, step=1)
                transport = st.selectbox("Moda Kendaraan", ["Motor", "Mobil", "Transportasi Umum"])
            with grid_col2:
                people = st.number_input("Jumlah Anggota (Orang)", min_value=1, max_value=10, value=2, step=1)
                lodging = st.selectbox("Opsi Akomodasi", ["Tanpa Penginapan", "Hemat", "Standar", "Nyaman"])

            budget = st.number_input("Alokasi Total Batas Anggaran (Rp)", min_value=100000, value=1500000, step=50000, format="%d")

            category = st.selectbox(
                "Kategori Tambahan",
                ["-"] + sorted(destination_base["kategori_wisata"].dropna().unique().tolist())
            )

            aktivitas_options = ["Foto-foto", "Kuliner", "Edukasi", "Keluarga", "Healing", "Belanja", "Sejarah", "Santai"]
            selected_activities = []
            
            cols_activities = st.columns(2)
            for i, activity in enumerate(aktivitas_options):
                col_idx = i % 2
                with cols_activities[col_idx]:
                    if st.checkbox(activity.capitalize(), key=f"act_it_{activity}"):
                        selected_activities.append(activity)

            st.markdown("<br>", unsafe_allow_html=True)
            run_itinerary = st.button("🔮 Buat Rencana", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            if run_itinerary:
                metrics["itinerary_request_total"].inc()
                if not selected_activities:
                    st.markdown(
                        '<div class="warning-pop">⚠️ Error: Silakan pilih minimal 1 jenis preferensi aktivitas pariwisata.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    jarak_hitung = float(dest.get('jarak', 0) or 0) if dest.get('jarak_tersedia', False) else 0
                    cost = estimate_budget_with_distance(days, people, transport, lodging, int(budget), jarak_hitung)

                    activities_str = " ".join(selected_activities)
                    query = f"{activities_str} {transport} {lodging}"
                    if category != "-":
                        query += f" {category}"

                    if dest.get('lat') is not None and dest.get('lon') is not None:
                        nearby_dest = recommend_by_query(query, top_n=10, user_location={'lat': dest['lat'], 'lon': dest['lon']})
                    else:
                        nearby_dest = recommend_by_query(query, top_n=10)

                    st.markdown('<div class="bounce-toast-success">🚀 Rute anggaran berhasil dihitung!</div>', unsafe_allow_html=True)

                    st.markdown(f"Itinerary untuk Perjalanan {days} Hari")
                    
                    c_biaya1, c_biaya2 = st.columns([1.2, 1])
                    with c_biaya1:
                        st.metric("Estimasi Total Biaya", rupiah(cost["total"]), help="Akumulasi seluruh komponen harian")
                    with c_biaya2:
                        if cost["total"] <= budget:
                            st.markdown('<div class="upload-pop" style="margin:0; padding:10px 14px; text-align:center;">✓ Sesuai Budget</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-pop" style="margin:0; padding:10px 14px; text-align:center;">⚠️ Melebihi Budget</div>', unsafe_allow_html=True)
                    
                    if dest.get('jarak_tersedia', False):
                        st.markdown(f'<div class="badge" style="margin-top:10px;">PP Kendaraan: {cost["jarak_tempuh"]:.1f} km</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("Rincian Kunjungan dan Perkiraan Biaya")

                    biaya_makan_per_hari = cost["makan"] / days
                    biaya_tiket_per_hari = cost["tiket"] / days
                    biaya_trans_hari_1 = cost["transportasi"]
                    biaya_trans_hari_lain = 25000 * people if transport != "Transportasi Umum" else 0
                    biaya_hotel_per_malam = cost["penginapan"] / max(days - 1, 1) if lodging != "Tanpa Penginapan" else 0

                    for d in range(days):
                        hari_ke = d + 1
                        st.markdown(f"""
                        <div style="border-left: 3px solid #0E9F8D; padding-left: 15px; margin-bottom: 15px; margin-top: 15px;">
                            <h5 style="color: #102A43; margin: 0 0 5px 0; font-size:15.5px;">📅 HARI KE-{hari_ke}</h5>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        idx_wisata_1 = d * 2
                        idx_wisata_2 = (d * 2) + 1
                        
                        if idx_wisata_1 < len(nearby_dest):
                            w1 = nearby_dest.iloc[idx_wisata_1]
                            st.markdown(f"**🏛️ Kunjungan Pagi:** {safe_text(w1.get('nama_wisata', '-')).title()} *({safe_text(w1.get('kategori_wisata', '-')).title()})*")
                        if idx_wisata_2 < len(nearby_dest):
                            w2 = nearby_dest.iloc[idx_wisata_2]
                            st.markdown(f"**🌳 Kunjungan Sore:** {safe_text(w2.get('nama_wisata', '-')).title()} *({safe_text(w2.get('kategori_wisata', '-')).title()})*")
                        
                        st.markdown('<div style="height:5px;"></div>', unsafe_allow_html=True)
                        
                        trans_hari_ini = biaya_trans_hari_1 if hari_ke == 1 else biaya_trans_hari_lain
                        hotel_hari_ini = biaya_hotel_per_malam if (hari_ke < days and lodging != "Tanpa Penginapan") else 0
                        total_hari_ini = biaya_makan_per_hari + biaya_tiket_per_hari + trans_hari_ini + hotel_hari_ini
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric(f"🍔 Konsumsi & Tiket (H{hari_ke})", rupiah(biaya_makan_per_hari + biaya_tiket_per_hari))
                        with m_col2:
                            st.metric(f"🚗 Logistik (H{hari_ke})", rupiah(trans_hari_ini + hotel_hari_ini))
                        with m_col3:
                            st.metric(f"📉 Subtotal H{hari_ke}", rupiah(total_hari_ini))
                            
                        st.markdown("<hr style='margin: 12px 0; border-style: dashed;'>", unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div class="card" style="text-align:center; padding: 40px 20px;">
                        <span style="font-size:40px;">📊</span>
                        <h4 style="margin-top:10px; color:#102A43;">Rencana akan dibuat otomatis!</h4>
                        <p class="muted" style="max-width:440px; margin: 0 auto;">
                            Isi preferensi di sebelah kiri, lalu sistem akan menampilkan estimasi budget dan daftar destinasi yang cocok!
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ==========================================
# 9. FEATURE LAYER 3: TENTANG APLIKASI
# ==========================================
elif selected_menu == "👥 Tentang Aplikasi":
    st.markdown('<div class="section-title">Tentang FURAP Travel Planner</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown(
            """
<div class="result-card">
    <div class="result-title" style="font-size:22px;">Integrasi Dua Model AI untuk Optimalisasi Sektor Pariwisata</div>
    <p style="font-size:14px; line-height:1.6; color:#334155;">
        FURAP Travel Planner merupakan sebuah platform yang dirancang untuk penentu keputusan pariwisata wilayah Bandung Raya. 
        Menggunakan pendekatan arsitektur model Convolutional Neural Network (CNN MobileNetV2) untuk melakukan identifikasi spasial landmark 
        melalui berkas input citra gambar, serta model Content-Based Filtering menggunakan algoritma K-Nearest Neighbors (KNN) 
        dan pembobotan fitur TF-IDF untuk menghasilkan susunan itinerary rekomendasi cerdas.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="card">
    <b style="color:#102A43; font-size:16px;">Pipeline Alur Kerja Sistem Pengguna</b>
    <div style="height:10px;"></div>
    <div class="card-compact">
        <b>1. Image Tracking Visual Extraction</b>
        <p class="muted">Pengguna memasukkan berkas gambar, model mengekstraksi matriks piksel gambar untuk menentukan nama destinasi pariwisata terdaftar.</p>
    </div>
    <div class="card-compact">
        <b>2. State Anchor Synchronization</b>
        <p class="muted">Nama objek pariwisata yang berhasil diidentifikasi disinkronisasi ke dalam memori Session State aplikasi sebagai koordinat pusat.</p>
    </div>
    <div class="card-compact">
        <b>3. Anggaran & Content Recommendation Generation</b>
        <p class="muted">Matriks pembobotan preferensi teks dicocokkan dengan dataset utama lokal untuk menyusun klaster rute kunjungan harian pariwisata.</p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_b:
        total_destinations = len(destination_base) if isinstance(destination_base, pd.DataFrame) else 0
        total_classes = len(class_names)

        st.markdown("📊 Ringkasan Kapasitas Basis Data Sistem")

        m1, m2 = st.columns(2)
        m1.metric("Destinasi Terdaftar", total_destinations)
        m2.metric("Klaster Gambar Citra", total_classes)

        st.markdown(
            """
<hr>
<p class="muted" style="font-size:12.5px;">
Artifacts model kecerdasan buatan berjalan dalam mode local caching memory, menjamin efisiensi latensi komputasi inferensi di bawah 1.5 detik per request pipeline.
</p>
""",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
<div class="card">
    <b>ℹ️ Catatan Teknis (Developer Note)</b>
    <p class="muted" style="font-size:12.5px; margin-top:5px;">
        Konfigurasi internal, status dependensi pustaka, parameter fungsional server web port 8501, serta pipeline 
        pengambilan metrik monitoring prometheus pada port 8000 berjalan di balik latar sistem pengguna untuk menjaga skalabilitas operasional prototipe.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("📂 Lihat Cakupan 31 Destinasi Bandung Raya yang Tersedia"):
            if isinstance(destination_base, pd.DataFrame) and "nama_wisata" in destination_base.columns:
                destination_list = (
                    destination_base["nama_wisata"]
                    .dropna()
                    .drop_duplicates()
                    .astype(str)
                    .sort_values()
                    .tolist()
                )
                for item in destination_list:
                    st.write(f"- {item.title()}")
            else:
                st.write("Basis data pariwisata kosong.")


# DEVELOPER TEAM LAYOUT GRID
    st.markdown("---")
    st.markdown('<div class="section-title">👥 Tim Pengembang Proyek Akhir FURAP-AI</div>', unsafe_allow_html=True)
    
    from pathlib import Path

    PLACEHOLDER_AVATAR = "https://api.dicebear.com/7.x/initials/svg?seed="

    team_members = [
        {
            "name": "Naufal Aji Ferdiansyah",
            "role": "Model Evaluation Engineer",
            "university": "Universitas Gunadarma",
            "photo": Path("assets/team/naufal.jpeg")
        },
        {
            "name": "Reza Rahmawati",
            "role": "ML Engineer",
            "university": "Universitas Terbuka",
            "photo": Path("assets/team/echa.png")
        },
        {
            "name": "Muhammad Dzaky Syamhaidar",
            "role": "Project Manager & MLOps Engineer",
            "university": "Universitas Gunadarma",
            "photo": Path("assets/team/syam.png")
        },
        {
            "name": "Diva Chintya Putri Nussandy",
            "role": "Data Engineer",
            "university": "UPN Veteran",
            "photo": Path("assets/team/diva.png")
        },
        {
            "name": "Devriean Maydhani Putro",
            "role": "MLOps & AI Deployment Engineer",
            "university": "Universitas Gunadarma",
            "photo": Path("assets/team/dev.png")
        }
    ]

    cols = st.columns(len(team_members))

    for col, member in zip(cols, team_members):
        with col:
            photo_path = member["photo"]

            if isinstance(photo_path, Path) and photo_path.exists():
                img = Image.open(photo_path).convert("RGB")
                img = img.resize((200, 200))

                import io
                import base64
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                b64 = base64.b64encode(buffer.getvalue()).decode()
                img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:50%; aspect-ratio:1/1; object-fit:cover; border:3px solid #0E9F8D;">'
            else:
                initials = "+".join(w[0] for w in member["name"].split() if w)
                avatar_url = PLACEHOLDER_AVATAR + initials
                img_tag = f'<img src="{avatar_url}" style="width:100%; border-radius:50%; aspect-ratio:1/1; object-fit:cover; border:3px solid #0E9F8D;">'

            st.markdown(
                f"""
<div class="card" style="text-align:center; padding:18px 8px; height:350px; display:flex; flex-direction:column; justify-content:space-between; align-items:center; box-sizing:border-box; overflow:hidden;">
    <div style="width:95px; height:95px; margin:0 auto 12px auto; overflow:hidden; border-radius:50%;">
        {img_tag}
    </div>
    <div style="font-weight:800; font-size:13.5px; color:#102A43; line-height:1.2;">{member['name']}</div>
    <div class="muted" style="font-size:11.5px; margin:4px 0;">{member['role']}</div>
    <div class="badge" style="font-size:10px; margin-top:6px; background:#F1F5F9; color:#475569; padding:4px 8px;">{member['university']}</div>
</div>
""",
                unsafe_allow_html=True,
            )

# GLOBAL PAGE FOOTER
st.markdown(
    """
<div class="footer">
    FURAP Travel Planner · AI Engineer at Pijak X IBM SkillsBuild 2026
</div>
""",
    unsafe_allow_html=True,
)