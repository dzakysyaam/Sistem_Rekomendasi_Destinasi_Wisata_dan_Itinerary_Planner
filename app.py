from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="FURAP Travel Planner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

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
except Exception as error:
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
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_place(image: Image.Image, top_n: int = 5):
    processed = preprocess_image(image)
    prediction = cnn_model.predict(processed, verbose=0)[0]

    top_indices = prediction.argsort()[-top_n:][::-1]
    top_predictions = []

    for idx in top_indices:
        class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        top_predictions.append({
            "class_name": class_name,
            "confidence": float(prediction[idx]),
        })

    best = top_predictions[0]
    return best["class_name"], best["confidence"], top_predictions

def recommend_by_query(query: str, top_n: int = 5, exclude_class: str | None = None) -> pd.DataFrame:
    query_vec = tfidf.transform([query])
    n_neighbors = min(top_n + 1, len(destination_base))
    distances, indices = content_knn.kneighbors(query_vec, n_neighbors=n_neighbors)

    result = destination_base.iloc[indices[0]].copy()
    result["similarity_score"] = 1 - distances[0]

    if exclude_class:
        result = result[result["class_name"] != exclude_class]

    return result.head(top_n).reset_index(drop=True)

def get_destination_detail(class_name: str) -> pd.DataFrame:
    matched = destination_base[destination_base["class_name"] == class_name]
    if not matched.empty:
        return matched.head(1)

    query = pretty_name(class_name)
    return recommend_by_query(query, top_n=1)

def estimate_budget(days: int, people: int, transport: str, lodging: str, budget: int) -> dict:
    meal_cost = days * people * 75000
    ticket_cost = days * people * 30000

    transport_rate = {
        "Motor": 60000,
        "Mobil": 200000,
        "Transportasi Umum": 120000,
    }.get(transport, 120000)

    lodging_rate = {
        "Tanpa Penginapan": 0,
        "Hemat": 180000,
        "Standar": 350000,
        "Nyaman": 550000,
    }.get(lodging, 0)

    transport_cost = transport_rate * days
    lodging_cost = lodging_rate * max(days - 1, 0)
    total = meal_cost + ticket_cost + transport_cost + lodging_cost

    return {
        "makan": meal_cost,
        "tiket": ticket_cost,
        "transportasi": transport_cost,
        "penginapan": lodging_cost,
        "total": total,
        "status": "Masih sesuai budget" if total <= budget else "Melebihi budget",
    }

st.markdown(
    """
<style>
:root {
    --navy: #102A43;
    --teal: #0E9F8D;
    --teal-dark: #04786C;
    --sand: #FFF7ED;
    --cream: #FFFBF4;
    --orange: #F59E0B;
    --muted: #64748B;
    --line: #E2E8F0;
    --card: rgba(255,255,255,0.88);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(14,159,141,0.18), transparent 32%),
        radial-gradient(circle at top right, rgba(245,158,11,0.14), transparent 26%),
        linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.72);
    backdrop-filter: blur(10px);
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 36px;
    border-radius: 30px;
    background:
        linear-gradient(135deg, rgba(16,42,67,0.98), rgba(4,120,108,0.92)),
        url("https://images.unsplash.com/photo-1500530855697-b586d89ba3ee");
    background-size: cover;
    background-position: center;
    color: white;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    margin-bottom: 22px;
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.18), transparent 20%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-label {
    display: inline-flex;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.24);
    font-size: 13px;
    letter-spacing: .4px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 44px;
    line-height: 1.05;
    font-weight: 850;
    margin: 0 0 12px 0;
}

.hero-subtitle {
    font-size: 16px;
    max-width: 720px;
    line-height: 1.7;
    opacity: .92;
}

.card {
    background: var(--card);
    border: 1px solid rgba(226,232,240,0.9);
    padding: 22px;
    border-radius: 24px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}

.card-compact {
    background: white;
    border: 1px solid var(--line);
    padding: 18px;
    border-radius: 20px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
    margin-bottom: 14px;
}

.result-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid var(--line);
    padding: 24px;
    border-radius: 28px;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.1);
    margin-bottom: 16px;
}

.section-title {
    color: var(--navy);
    font-size: 26px;
    font-weight: 850;
    margin: 6px 0 14px 0;
}

.result-title {
    color: var(--navy);
    font-size: 30px;
    line-height: 1.2;
    font-weight: 850;
    margin-bottom: 10px;
}

.badge {
    display: inline-flex;
    width: fit-content;
    padding: 8px 13px;
    border-radius: 999px;
    background: #DFF7F1;
    color: var(--teal-dark);
    font-weight: 750;
    font-size: 13px;
    margin: 6px 8px 10px 0;
}

.badge-soft {
    display: inline-flex;
    width: fit-content;
    padding: 8px 13px;
    border-radius: 999px;
    background: var(--sand);
    color: #9A5B00;
    font-weight: 750;
    font-size: 13px;
    margin: 6px 8px 10px 0;
}

.muted {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
}

.upload-pop {
    padding: 14px 16px;
    border-radius: 16px;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #047857;
    font-weight: 700;
    margin: 12px 0;
    box-shadow: 0 10px 24px rgba(4,120,87,0.08);
}

.warning-pop {
    padding: 14px 16px;
    border-radius: 16px;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    color: #92400E;
    font-weight: 700;
    margin: 12px 0;
}

div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, var(--teal), var(--teal-dark));
    color: white;
    border: none;
    border-radius: 16px;
    padding: 12px 18px;
    font-weight: 800;
    box-shadow: 0 10px 22px rgba(14,159,141,0.22);
}

div.stButton > button:hover, div.stDownloadButton > button:hover {
    color: white;
    border: none;
    transform: translateY(-1px);
}

[data-testid="stTabs"] button {
    font-weight: 800;
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-weight: 850;
}

hr {
    border-color: var(--line);
}

.footer {
    color: var(--muted);
    text-align: center;
    font-size: 13px;
    margin-top: 28px;
}
</style>
""",
    unsafe_allow_html=True,
)

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

tab_image, tab_itinerary, tab_about = st.tabs([
    "Tempat Wisata",
    "Rencana Perjalanan",
    "Tentang Aplikasi",
])


with tab_image:
    st.markdown('<div class="section-title">Kenali destinasi dari gambar</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Unggah gambar destinasi")
        st.caption("Gunakan foto yang jelas dan menampilkan ciri khas tempat.")

        uploaded_file = st.file_uploader(
            "Pilih gambar",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

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
                st.toast("Gambar berhasil diunggah.")

            st.markdown(
                '<div class="upload-pop">Gambar berhasil diunggah. Klik tombol analisis untuk melihat hasil.</div>',
                unsafe_allow_html=True
            )

            image = Image.open(image_file)
            st.image(image, caption="Pratinjau gambar", use_container_width=True)
        else:
            image = None

        run_image = st.button("Analisis Gambar", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
<div class="card-compact">
    <b>Tips gambar yang bagus</b>
    <p class="muted">
        Pilih foto yang tidak blur, cukup terang, dan memperlihatkan landmark utama seperti tulisan lokasi,
        bangunan khas, gerbang, area depan, atau ikon tempat.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        if run_image:
            if image is None:
                st.markdown(
                    '<div class="warning-pop">Silakan unggah gambar terlebih dahulu.</div>',
                    unsafe_allow_html=True
                )
            else:
                with st.spinner("Sedang membaca gambar..."):
                    predicted_class, confidence, top_predictions = predict_place(image)
                    detail = get_destination_detail(predicted_class)

                st.toast("Analisis selesai.")

                if not detail.empty:
                    row = detail.iloc[0]
                    nama_wisata = safe_text(row.get("nama_wisata", pretty_name(predicted_class))).title()
                    kategori = safe_text(row.get("kategori_wisata", "-")).title()
                    deskripsi = safe_text(row.get("deskripsi_wisata", "Deskripsi belum tersedia."))
                    alamat = safe_text(row.get("alamat", "-"))

                    st.markdown(
                        f"""
<div class="result-card">
    <div class="result-title">{nama_wisata}</div>
    <div class="badge">{kategori}</div>
    <div class="badge-soft">Keyakinan sistem: {confidence * 100:.2f}%</div>
    <p style="margin-top: 12px;">{deskripsi}</p>
    <p class="muted"><b>Alamat:</b> {alamat}</p>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    if confidence < 0.60:
                        st.markdown(
                            '<div class="warning-pop">Hasil belum terlalu kuat. Gunakan gambar yang lebih jelas atau menampilkan ikon utama destinasi.</div>',
                            unsafe_allow_html=True
                        )

                    maps_url = maps_link(row)
                    st.link_button("Buka lokasi", maps_url, use_container_width=True)

                    if pd.notna(row.get("latitude", None)) and pd.notna(row.get("longitude", None)):
                        st.map(
                            pd.DataFrame([{"lat": row["latitude"], "lon": row["longitude"]}]),
                            zoom=12,
                            use_container_width=True
                        )

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Kemungkinan lain")
                    top_df = pd.DataFrame([
                        {
                            "Destinasi": pretty_name(item["class_name"]),
                            "Keyakinan": f"{item['confidence'] * 100:.2f}%"
                        }
                        for item in top_predictions
                    ])
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### Rekomendasi serupa")

                    recs = recommend_by_query(
                        row.get("teks_destinasi", predicted_class),
                        top_n=4,
                        exclude_class=row.get("class_name")
                    )

                    if recs.empty:
                        st.info("Rekomendasi serupa belum tersedia.")
                    else:
                        for _, rec in recs.iterrows():
                            st.markdown(
                                f"""
<div class="card-compact">
    <b>{safe_text(rec.get('nama_wisata', '-')).title()}</b>
    <div class="muted">{safe_text(rec.get('kategori_wisata', '-')).title()} · Kecocokan {float(rec.get('similarity_score', 0)) * 100:.1f}%</div>
    <p>{safe_text(rec.get('deskripsi_wisata', ''))}</p>
</div>
""",
                                unsafe_allow_html=True,
                            )

                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"""
<div class="result-card">
    <div class="result-title">{pretty_name(predicted_class)}</div>
    <div class="badge-soft">Keyakinan sistem: {confidence * 100:.2f}%</div>
    <p>Detail destinasi belum tersedia di data referensi.</p>
</div>
""",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                """
<div class="card">
    <div class="section-title">Hasil akan tampil di sini</div>
    <p class="muted">
        Setelah gambar diunggah dan dianalisis, halaman ini akan menampilkan nama destinasi,
        tingkat keyakinan, deskripsi, lokasi, dan rekomendasi tempat serupa.
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
                    st.map(map_df, zoom=8, use_container_width=True)

with tab_itinerary:
    st.markdown('<div class="section-title">Susun rencana perjalanan</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Preferensi perjalanan")

        category = st.selectbox(
            "Kategori wisata",
            sorted(destination_base["kategori_wisata"].dropna().unique().tolist())
        )

        activity = st.selectbox(
            "Aktivitas",
            ["foto-foto", "kuliner", "edukasi", "keluarga", "healing", "belanja", "sejarah", "santai"]
        )

        days = st.number_input("Durasi perjalanan", min_value=1, max_value=5, value=2, help="Dalam hari")
        people = st.number_input("Jumlah orang", min_value=1, max_value=10, value=2)
        budget = st.number_input("Total budget", min_value=100000, value=1500000, step=50000)
        transport = st.selectbox("Transportasi", ["Motor", "Mobil", "Transportasi Umum"])
        lodging = st.selectbox("Penginapan", ["Tanpa Penginapan", "Hemat", "Standar", "Nyaman"])
        origin = st.text_input("Lokasi awal", value="Jakarta")

        run_itinerary = st.button("Buat Rencana", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if run_itinerary:
            query = f"{category} {activity} {transport} {lodging} {origin}"
            recs = recommend_by_query(query, top_n=5)
            cost = estimate_budget(days, people, transport, lodging, int(budget))

            st.toast("Rencana perjalanan berhasil dibuat.")

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### Ringkasan budget")

            b1, b2, b3 = st.columns(3)
            b1.metric("Makan", rupiah(cost["makan"]))
            b2.metric("Transport", rupiah(cost["transportasi"]))
            b3.metric("Total", rupiah(cost["total"]))

            b4, b5 = st.columns(2)
            b4.metric("Tiket", rupiah(cost["tiket"]))
            b5.metric("Penginapan", rupiah(cost["penginapan"]))

            if cost["total"] <= budget:
                st.markdown('<div class="upload-pop">Estimasi biaya masih sesuai dengan budget.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-pop">Estimasi biaya melebihi budget. Pertimbangkan transportasi atau penginapan yang lebih hemat.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Destinasi yang cocok")

            for idx, row in recs.iterrows():
                st.markdown(
                    f"""
<div class="card-compact">
    <b>{idx + 1}. {safe_text(row.get('nama_wisata', '-')).title()}</b>
    <div class="muted">{safe_text(row.get('kategori_wisata', '-')).title()} · Kecocokan {float(row.get('similarity_score', 0)) * 100:.1f}%</div>
    <p>{safe_text(row.get('deskripsi_wisata', ''))}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                """
<div class="card">
    <div class="section-title">Rencana akan dibuat otomatis</div>
    <p class="muted">
        Isi preferensi di sebelah kiri, lalu sistem akan menampilkan estimasi budget dan daftar destinasi yang cocok.
    </p>
</div>
""",
                unsafe_allow_html=True,
            )


with tab_about:
    st.markdown('<div class="section-title">Tentang FURAP Travel Planner</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown(
            """
<div class="result-card">
    <div class="result-title">Travel planning yang lebih praktis dan nyaman.</div>
    <p>
        FURAP Travel Planner membantu pengguna mengenali destinasi wisata dari gambar
        dan menyusun gambaran rencana perjalanan berdasarkan preferensi yang dipilih.
    </p>
    <p class="muted">
        Aplikasi ini dirancang untuk memberikan pengalaman yang sederhana:
        pengguna cukup mengunggah gambar atau mengisi preferensi perjalanan,
        lalu sistem menampilkan hasil yang mudah dipahami.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="card">
    <b>Alur penggunaan</b>
    <div style="height:10px;"></div>
    <div class="card-compact">
        <b>1. Kenali destinasi</b>
        <p class="muted">Unggah foto tempat wisata, lalu aplikasi akan menampilkan nama destinasi, deskripsi, lokasi, dan pilihan wisata serupa.</p>
    </div>
    <div class="card-compact">
        <b>2. Buat rencana perjalanan</b>
        <p class="muted">Isi kategori wisata, durasi, jumlah orang, budget, transportasi, dan penginapan untuk melihat estimasi perjalanan.</p>
    </div>
    <div class="card-compact">
        <b>3. Bandingkan rekomendasi</b>
        <p class="muted">Lihat beberapa destinasi yang cocok agar pengguna bisa memilih rencana perjalanan yang paling sesuai.</p>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_b:
        total_destinations = len(destination_base) if isinstance(destination_base, pd.DataFrame) else 0
        total_classes = len(class_names)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Ringkasan aplikasi")

        m1, m2 = st.columns(2)
        m1.metric("Destinasi", total_destinations)
        m2.metric("Kategori gambar", total_classes)

        st.markdown(
            """
<hr>
<p class="muted">
Aplikasi ini cocok digunakan sebagai prototipe rekomendasi wisata,
demo capstone, dan pengujian awal pengalaman pengguna.
</p>
""",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
<div class="card">
    <b>Developer Note</b>
    <p class="muted">
        Halaman ini sengaja dibuat ringkas agar pengguna tidak melihat detail teknis internal.
        Informasi teknis seperti nama file model, status artifact, dan konfigurasi sistem
        tidak ditampilkan pada tampilan pengguna.
    </p>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("Daftar destinasi yang tersedia"):
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
                st.write("Data destinasi belum tersedia.")

st.markdown(
    """
<div class="footer">
    FURAP Travel Planner · Sistem rekomendasi wisata dan rencana perjalanan
</div>
""",
    unsafe_allow_html=True,
)
