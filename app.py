import streamlit as st

# =========================================================
# PAGE CONFIG
# HARUS MENJADI PERINTAH STREAMLIT PERTAMA
# =========================================================

st.set_page_config(
    page_title="ChiliGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# IMPORT LIBRARY
# =========================================================

import cv2
import numpy as np
import joblib

from skimage.feature import graycomatrix, graycoprops


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"


# =========================================================
# LOAD MODEL DAN SCALER
# =========================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model()

except Exception as e:

    st.error("❌ Model atau scaler gagal dimuat.")

    st.code(str(e))

    st.stop()


# =========================================================
# EKSTRAKSI 6 FITUR HSV
# =========================================================

def extract_hsv_features(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    features = [

        np.mean(h),
        np.mean(s),
        np.mean(v),

        np.std(h),
        np.std(s),
        np.std(v)

    ]

    return features


# =========================================================
# EKSTRAKSI 5 FITUR GLCM
#
# HARUS KONSISTEN DENGAN TRAINING:
#
# grayscale
# -> /4
# -> 64 level
# -> distance = 1
# -> angle = 0
#
# =========================================================

def extract_glcm_features(image):

    # BGR -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Resize sama dengan pipeline
    gray = cv2.resize(
        gray,
        (224, 224)
    )

    # -----------------------------------------------------
    # KUANTISASI
    # 256 grayscale -> 64 level
    # -----------------------------------------------------

    gray = (gray / 4).astype(np.uint8)

    # -----------------------------------------------------
    # GLCM
    # -----------------------------------------------------

    glcm = graycomatrix(
        gray,

        distances=[1],

        angles=[0],

        levels=64,

        symmetric=True,

        normed=True
    )

    # -----------------------------------------------------
    # 5 FITUR GLCM
    # -----------------------------------------------------

    contrast = graycoprops(
        glcm,
        "contrast"
    )[0, 0]

    dissimilarity = graycoprops(
        glcm,
        "dissimilarity"
    )[0, 0]

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    )[0, 0]

    energy = graycoprops(
        glcm,
        "energy"
    )[0, 0]

    correlation = graycoprops(
        glcm,
        "correlation"
    )[0, 0]

    features = [

        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation

    ]

    return features


# =========================================================
# EKSTRAKSI TOTAL 11 FITUR
#
# 6 HSV + 5 GLCM = 11
# =========================================================

def extract_features(image):

    # Resize
    image = cv2.resize(
        image,
        (224, 224)
    )

    # 6 HSV
    hsv_features = extract_hsv_features(
        image
    )

    # 5 GLCM
    glcm_features = extract_glcm_features(
        image
    )

    # Gabungkan
    features = (
        hsv_features +
        glcm_features
    )

    return np.array(
        features,
        dtype=np.float64
    )


# =========================================================
# CEK MODEL
# =========================================================

try:

    expected_features = scaler.n_features_in_

except AttributeError:

    expected_features = 11


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GLOBAL
===================================================== */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =====================================================
   INTRO CARD
===================================================== */

.cg-card {

    padding: 25px;

    border-radius: 20px;

    border:
        1px solid
        rgba(34, 197, 94, 0.22);

    background:
        linear-gradient(
            145deg,
            rgba(34, 197, 94, 0.08),
            rgba(0, 0, 0, 0.02)
        );

    box-shadow:
        0 10px 30px
        rgba(0, 0, 0, 0.06);

    margin-bottom: 25px;
}


/* =====================================================
   SECTION TITLE
===================================================== */

.section-title {

    font-size: 21px;

    font-weight: 750;

    margin-top: 8px;

    margin-bottom: 12px;
}


/* =====================================================
   RESULT CARD
===================================================== */

.result-card {

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    border:
        1px solid
        rgba(34, 197, 94, 0.30);

    background:
        linear-gradient(
            145deg,
            rgba(34, 197, 94, 0.13),
            rgba(34, 197, 94, 0.025)
        );

    box-shadow:
        0 12px 35px
        rgba(34, 197, 94, 0.08);
}


.result-label {

    font-size: 12px;

    text-transform: uppercase;

    letter-spacing: 1.5px;

    color: #6b7280;

    margin-bottom: 8px;
}


.result-name {

    font-size: 30px;

    font-weight: 800;

    color: #16a34a;

    margin-bottom: 10px;
}


.result-confidence {

    font-size: 17px;

    font-weight: 600;
}


/* =====================================================
   METRIC CARD
===================================================== */

.metric-box {

    padding: 18px;

    border-radius: 16px;

    text-align: center;

    border:
        1px solid
        rgba(34, 197, 94, 0.18);

    background:
        rgba(34, 197, 94, 0.05);

    min-height: 85px;
}


.metric-value {

    font-size: 23px;

    font-weight: 800;

    color: #16a34a;
}


.metric-label {

    font-size: 11px;

    color: #6b7280;

    margin-top: 6px;

    line-height: 1.3;
}


/* =====================================================
   SIDEBAR
===================================================== */

[data-testid="stSidebar"] {

    border-right:
        1px solid
        rgba(34, 197, 94, 0.15);
}


/* =====================================================
   UPLOADER
===================================================== */

[data-testid="stFileUploader"] {

    border-radius: 15px;
}


/* =====================================================
   PROGRESS BAR
===================================================== */

.stProgress > div > div > div > div {

    background-color: #22c55e;

}


/* =====================================================
   DARK MODE
===================================================== */

@media (prefers-color-scheme: dark) {

    .cg-card {

        background:
            linear-gradient(
                145deg,
                rgba(34, 197, 94, 0.10),
                rgba(0, 0, 0, 0.28)
            );

        border-color:
            rgba(34, 197, 94, 0.22);

        box-shadow:
            0 10px 30px
            rgba(0, 0, 0, 0.30);
    }


    .result-card {

        background:
            linear-gradient(
                145deg,
                rgba(34, 197, 94, 0.16),
                rgba(0, 0, 0, 0.30)
            );

        border-color:
            rgba(34, 197, 94, 0.30);

        box-shadow:
            0 12px 40px
            rgba(0, 0, 0, 0.35);
    }


    .result-name {

        color: #4ade80;

    }


    .metric-box {

        background:
            rgba(34, 197, 94, 0.08);

        border-color:
            rgba(34, 197, 94, 0.18);

    }


    .metric-value {

        color: #4ade80;

    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# PAKAI STREAMLIT NATIVE
# =========================================================

st.markdown(
    "# 🌿 ChiliGuard"
)

st.markdown(
    "**AI-based Chili Leaf Disease Classification**"
)

st.caption(
    "HSV + GLCM Feature Extraction • Support Vector Machine"
)

st.write("")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌿 ChiliGuard")

    st.caption(
        "Sistem klasifikasi citra penyakit "
        "daun cabai berbasis "
        "Support Vector Machine."
    )

    st.divider()

    st.markdown("### 🧠 Model")

    st.write("**Algorithm**")

    st.caption(
        "Support Vector Machine"
    )

    st.write("**Kernel**")

    st.caption(
        "RBF"
    )

    st.write("**C**")

    st.caption(
        "10"
    )

    st.write("**Features**")

    st.caption(
        "HSV + GLCM"
    )

    st.write("**Total Features**")

    st.caption(
        "11 Features"
    )

    st.divider()

    st.markdown("### 🌱 Classes")

    st.write(
        "🟢 Anthracnose"
    )

    st.write(
        "🟢 Cercospora Leaf Spot"
    )

    st.write(
        "🟢 Fresh Leaf"
    )

    st.write(
        "🟢 Leaf Curl Disease"
    )

    st.divider()

    st.caption(
        "ChiliGuard Research Project"
    )


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("""
<div class="cg-card">

<h3>🔍 Deteksi Penyakit Daun Cabai</h3>

<p>
Upload foto daun cabai untuk mendapatkan
hasil klasifikasi menggunakan model
<b>Support Vector Machine (SVM)</b>.
</p>

<p>
Sistem menganalisis karakteristik warna
<b>HSV</b> dan karakteristik tekstur
<b>GLCM</b> dari citra.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📷 Upload Foto Daun Cabai'
    '</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Pilih foto daun cabai",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # READ IMAGE
    # =====================================================

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    # =====================================================
    # VALIDATE IMAGE
    # =====================================================

    if image is None:

        st.error(
            "❌ Gambar tidak dapat dibaca."
        )

        st.stop()


    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # EXTRACT 11 FEATURES
    # =====================================================

    features = extract_features(
        image
    )


    # Pastikan bentuknya 1 x 11
    features = features.reshape(
        1,
        -1
    )


    # =====================================================
    # FEATURE COUNT CHECK
    # =====================================================

    if features.shape[1] != expected_features:

        st.error(
            f"❌ Jumlah fitur tidak cocok. "
            f"GUI menghasilkan {features.shape[1]} fitur, "
            f"sedangkan scaler mengharapkan "
            f"{expected_features} fitur."
        )

        st.stop()


    # =====================================================
    # STANDARD SCALER
    # =====================================================

    try:

        features_scaled = scaler.transform(
            features
        )

    except Exception as e:

        st.error(
            "❌ Terjadi masalah pada StandardScaler."
        )

        st.code(
            str(e)
        )

        st.stop()


    # =====================================================
    # SVM PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            features_scaled
        )[0]


        probabilities = model.predict_proba(
            features_scaled
        )[0]

    except Exception as e:

        st.error(
            "❌ Terjadi masalah saat prediksi SVM."
        )

        st.code(
            str(e)
        )

        st.stop()


    # =====================================================
    # CLASS INFORMATION
    # =====================================================

    classes = list(
        model.classes_
    )


    predicted_index = classes.index(
        prediction
    )


    confidence = probabilities[
        predicted_index
    ]


    # =====================================================
    # IMAGE + RESULT
    # =====================================================

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # =====================================================
    # IMAGE
    # =====================================================

    with col1:

        st.markdown(
            '<div class="section-title">'
            '🖼️ Input Image'
            '</div>',
            unsafe_allow_html=True
        )


        st.image(
            image_rgb,
            use_container_width=True
        )


    # =====================================================
    # RESULT
    # =====================================================

    with col2:

        st.markdown(
            '<div class="section-title">'
            '🎯 Classification Result'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
<div class="result-card">

<div class="result-label">
Predicted Class
</div>

<div class="result-name">
{prediction}
</div>

<div class="result-confidence">
Confidence: {confidence * 100:.2f}%
</div>

</div>
""",
            unsafe_allow_html=True
        )


        st.write("")


        st.progress(
            float(confidence)
        )


    # =====================================================
    # PROBABILITY SUMMARY
    # =====================================================

    st.write("")


    st.markdown(
        '<div class="section-title">'
        '📊 Prediction Probability'
        '</div>',
        unsafe_allow_html=True
    )


    prob_cols = st.columns(
        len(classes)
    )


    for col, class_name, probability in zip(
        prob_cols,
        classes,
        probabilities
    ):

        with col:

            st.markdown(
                f"""
<div class="metric-box">

<div class="metric-value">
{probability * 100:.1f}%
</div>

<div class="metric-label">
{class_name}
</div>

</div>
""",
                unsafe_allow_html=True
            )


    # =====================================================
    # PROBABILITY DETAIL
    # =====================================================

    st.write("")

    st.markdown(
        "### 📈 Detail Probabilitas"
    )


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        st.write(
            f"**{class_name}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # =====================================================
    # FEATURES
    # =====================================================

    st.write("")


    with st.expander(
        "🔬 Lihat 11 Fitur Citra"
    ):

        feature_names = [

            "H Mean",
            "S Mean",
            "V Mean",

            "H Std",
            "S Std",
            "V Std",

            "GLCM Contrast",
            "GLCM Dissimilarity",
            "GLCM Homogeneity",
            "GLCM Energy",
            "GLCM Correlation"

        ]


        feature_cols = st.columns(
            2
        )


        for i, (
            name,
            value
        ) in enumerate(
            zip(
                feature_names,
                features[0]
            )
        ):

            with feature_cols[
                i % 2
            ]:

                st.write(
                    f"**{name}:** "
                    f"{value:.6f}"
                )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.write("")


    with st.expander(
        "⚙️ Informasi Model"
    ):

        info_col1, info_col2, info_col3 = st.columns(3)


        with info_col1:

            st.metric(
                "Algorithm",
                "SVM"
            )


        with info_col2:

            st.metric(
                "Kernel",
                "RBF"
            )


        with info_col3:

            st.metric(
                "Features",
                "11"
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="cg-footer">

🌿 <b>ChiliGuard</b>

<br><br>

Sistem Klasifikasi Citra Penyakit Daun Cabai
Berbasis Support Vector Machine

<br><br>

HSV • GLCM • SVM

</div>
""", unsafe_allow_html=True)