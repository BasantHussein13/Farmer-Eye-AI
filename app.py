
import io
import sys
import time
import zipfile
from html import escape
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

BASE_DIR = Path(__file__).resolve().parent

TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from modules.prediction import (
    analyze_image,
    model as prediction_model,
    MODEL_NAMES,
)

from modules.gradcam import generate_gradcam

from modules.analytics import (
    get_prediction_history,
    get_general_statistics,
    get_disease_distribution,
    update_prediction_feedback,
)


# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="عين المزارع | Farmer Eye AI",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Arabic Labels
# ==============================================================================

STATUS_AR = {
    "Healthy": "سليمة",
    "Diseased": "مصابة",
    "Uncertain": "غير مؤكدة",
}

DISEASE_AR = {
    "Angular Leafspot": "التبقع الزاوي للأوراق",
    "Anthracnose Fruit Rot": "عفن الأنثراكنوز للثمار",
    "Blossom Blight": "لفحة الأزهار",
    "Gray Mold": "العفن الرمادي",
    "Healthy Strawberry": "فراولة سليمة",
    "Leaf Spot": "تبقع الأوراق",
    "Powdery Mildew Fruit": "البياض الدقيقي على الثمار",
    "Powdery Mildew Leaf": "البياض الدقيقي على الأوراق",
}

RECOMMENDATIONS_AR = {
    "Angular Leafspot":
        "اعزل الأوراق المصابة، قلل بلل الأوراق، وحافظ على تهوية جيدة مع متابعة تطور البقع.",
    "Anthracnose Fruit Rot":
        "أزل الثمار المصابة من الحقل، تجنب ملامسة الثمار للتربة، وحسن التهوية حول النباتات.",
    "Blossom Blight":
        "أزل الأزهار والأنسجة المتضررة وقلل الرطوبة حول النبات مع متابعة الأزهار الجديدة.",
    "Gray Mold":
        "أزل الأجزاء المصابة، قلل الرطوبة، وتجنب الري المباشر فوق الأوراق والثمار.",
    "Leaf Spot":
        "أزل الأوراق شديدة الإصابة، حسن التهوية، وراقب ظهور بقع جديدة على الأوراق.",
    "Powdery Mildew Fruit":
        "افصل الثمار المصابة وقلل الرطوبة الزائدة مع فحص باقي الثمار بشكل دوري.",
    "Powdery Mildew Leaf":
        "أزل الأوراق شديدة الإصابة وحسن حركة الهواء حول النباتات وراقب الأوراق الحديثة.",
    "Healthy Strawberry":
        "لا توجد علامات مرضية واضحة في الصورة. استمر في المتابعة الدورية للنبات.",
}

RISK_AR = {
    "Healthy": "منخفض",
    "Diseased": "يحتاج متابعة",
    "Uncertain": "غير محدد",
}


# ==============================================================================
# Styling
# ==============================================================================

st.markdown(
    """
    <style>

    :root {
        --berry:#e93672;
        --berry-dark:#c91f59;
        --berry-soft:#fff1f6;
        --berry-border:#f5c8d8;
        --green:#2eaa70;
        --green-soft:#edf9f3;
        --navy:#26364d;
        --ink:#222a35;
        --muted:#6c7480;
        --line:#eedde4;
        --card:#ffffff;
        --shadow:0 14px 38px rgba(66, 32, 47, .08);
    }

    html, body, [class*="css"] {
        direction: rtl;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, #ffe6ef 0, transparent 22%),
            linear-gradient(180deg, #fffafb 0%, #fff6f9 55%, #ffffff 100%);
        color:var(--ink);
    }

    .block-container {
        max-width:1320px;
        padding-top:1.2rem;
        padding-bottom:3rem;
    }

    section[data-testid="stSidebar"] {
        background:linear-gradient(180deg,#ffffff 0%,#fff8fa 100%);
        border-left:1px solid var(--line);
    }

    section[data-testid="stSidebar"] > div {
        padding-top:1rem;
    }

    .side-brand {
        text-align:center;
        background:white;
        border:1px solid var(--line);
        box-shadow:var(--shadow);
        border-radius:26px;
        padding:22px 14px;
        margin-bottom:16px;
    }

    .side-brand .logo {
        font-size:54px;
        margin-bottom:4px;
    }

    .side-brand h2 {
        margin:0;
        color:var(--berry-dark);
        font-weight:900;
        font-size:1.6rem;
    }

    .side-brand p {
        color:var(--muted);
        margin:7px 0 0;
        font-size:.85rem;
    }

    .hero {
        position:relative;
        overflow:hidden;
        border-radius:30px;
        padding:36px 40px;
        margin-bottom:24px;
        color:#fff;
        box-shadow:0 18px 48px rgba(210,35,96,.21);
        background:
            radial-gradient(circle at 13% 24%,rgba(255,255,255,.17),transparent 18%),
            radial-gradient(circle at 88% 12%,rgba(255,255,255,.13),transparent 20%),
            linear-gradient(135deg,#ef3e78 0%,#f65386 46%,#cf245f 100%);
    }

    .hero:before,
    .hero:after {
        content:"🍓";
        position:absolute;
        opacity:.17;
        font-size:100px;
    }

    .hero:before {
        left:24px;
        top:0;
        transform:rotate(-12deg);
    }

    .hero:after {
        right:34px;
        bottom:-28px;
        transform:rotate(15deg);
    }

    .hero h1 {
        font-size:2.75rem;
        margin:0;
        font-weight:900;
    }

    .hero p {
        max-width:770px;
        margin:12px 0 0;
        font-size:1.07rem;
        line-height:1.95;
        opacity:.98;
    }

    .hero-pill {
        display:inline-block;
        margin-top:16px;
        border-radius:999px;
        padding:8px 15px;
        background:rgba(255,255,255,.17);
        border:1px solid rgba(255,255,255,.28);
        font-weight:800;
    }

    .section-title {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:14px;
        margin:10px 0 14px;
    }

    .section-title h3 {
        margin:0;
        font-size:1.3rem;
        font-weight:900;
        color:#2b3340;
    }

    .subtle {
        color:var(--muted);
        font-size:.92rem;
    }

    .card {
        background:white;
        border:1px solid var(--line);
        border-radius:22px;
        box-shadow:var(--shadow);
        padding:20px;
    }

    .step-card {
        min-height:165px;
        background:white;
        border:1px solid var(--line);
        border-radius:22px;
        box-shadow:var(--shadow);
        padding:20px;
    }

    .step-num {
        display:inline-flex;
        width:36px;
        height:36px;
        border-radius:50%;
        align-items:center;
        justify-content:center;
        color:white;
        background:linear-gradient(135deg,var(--berry),var(--berry-dark));
        font-weight:900;
        margin-bottom:10px;
    }

    .step-card h4 {
        margin:4px 0 7px;
        font-size:1.1rem;
    }

    .step-card p {
        margin:0;
        color:var(--muted);
        line-height:1.75;
        font-size:.92rem;
    }

    .result-card {
        background:white;
        border:1px solid var(--line);
        border-top:5px solid var(--berry);
        border-radius:24px;
        box-shadow:var(--shadow);
        padding:24px;
        text-align:center;
    }

    .result-title {
        font-size:1.85rem;
        font-weight:900;
        margin:6px 0;
        color:var(--navy);
    }

    .result-disease {
        color:var(--berry-dark);
        font-weight:900;
        font-size:1.35rem;
        margin:6px 0 10px;
    }

    .confidence-pill {
        display:inline-block;
        padding:8px 14px;
        border-radius:999px;
        background:var(--berry-soft);
        border:1px solid var(--berry-border);
        color:var(--berry-dark);
        font-weight:900;
    }

    .action-card {
        margin-top:14px;
        padding:15px 16px;
        border-radius:16px;
        background:linear-gradient(135deg,#fffaf1,#fff5e7);
        border:1px solid #f2ddb4;
        color:#675024;
        line-height:1.85;
        text-align:right;
    }

    .status-good {
        color:var(--green);
        font-weight:900;
    }

    .status-bad {
        color:var(--berry-dark);
        font-weight:900;
    }

    .mini-badge {
        display:inline-block;
        padding:5px 10px;
        border-radius:999px;
        background:#f5f7fa;
        color:#58606d;
        font-size:.8rem;
        font-weight:700;
    }

    div[data-testid="stMetric"] {
        background:white;
        border:1px solid var(--line);
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 8px 20px rgba(40,25,34,.05);
    }

    div[data-testid="stMetricValue"] {
        color:var(--navy);
        font-weight:900;
    }

    div[data-testid="stFileUploader"] {
        background:white;
        border:1px dashed #ee8eae;
        border-radius:22px;
        padding:8px;
    }

    .stButton > button {
        width:100%;
        min-height:48px;
        border-radius:14px;
        border:none;
        color:white;
        font-weight:900;
        background:linear-gradient(135deg,var(--berry),var(--berry-dark));
        box-shadow:0 8px 18px rgba(217,47,104,.18);
    }

    .stButton > button:hover {
        color:white;
        border:none;
        transform:translateY(-1px);
    }

    .footer {
        text-align:center;
        color:#8b7480;
        border-top:1px solid var(--line);
        margin-top:30px;
        padding-top:16px;
        font-size:.85rem;
    }

    @media (max-width:900px) {
        .hero {padding:26px 22px;}
        .hero h1 {font-size:2rem;}
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Helpers
# ==============================================================================

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <h1>عين المزارع 🍓</h1>
            <p>
                افحص صورة الفراولة في ثوانٍ، واعرف إذا كانت سليمة أو مصابة،
                وشاهد مكان الإصابة واحصل على إرشاد مبسط للخطوة التالية.
            </p>
            <span class="hero-pill">فحص سريع • تشخيص بصري • سجل منظم • تحليلات ذكية</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    path = TEMP_DIR / f"{time.time_ns()}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def disease_name_ar(name):
    if (
        name is None
        or pd.isna(name)
        or not str(name).strip()
        or str(name).strip().lower() in {
            "none",
            "nan",
            "-"
        }
    ):
        return "-"

    diseases = [
        value.strip()
        for value in str(name).split(",")
        if value.strip()
    ]

    return "، ".join(
        DISEASE_AR.get(disease, disease)
        for disease in diseases
    )


def main_disease_name(result):
    name = result.get("disease_name")
    if not name:
        return (
            "Healthy Strawberry"
            if result.get("status") == "Healthy"
            else None
        )
    return name.split(",")[0].strip()


def recommendation_for(result):
    disease = main_disease_name(result)
    if disease is None:
        return (
            "أعد التصوير في إضاءة جيدة ومن مسافة أقرب، "
            "وتأكد أن الثمرة أو الورقة واضحة داخل الصورة."
        )

    return RECOMMENDATIONS_AR.get(
        disease,
        (
            "اعزل الأجزاء المشتبه بها وراقب تطور الأعراض، "
            "واستعن بمهندس زراعي إذا استمرت الإصابة أو توسعت."
        ),
    )


def confidence_label(value):
    value = float(value)

    if value >= 0.85:
        return "ثقة عالية"

    if value >= 0.70:
        return "ثقة جيدة"

    if value >= 0.50:
        return "ثقة متوسطة"

    return "ثقة منخفضة"


def existing_path(value):
    if value is None:
        return None

    try:
        path = Path(str(value))
        return path if path.exists() else None
    except Exception:
        return None



def build_printable_report_html(
    *,
    title,
    status,
    disease_name,
    confidence,
    detected_objects,
    analysis_date="-",
    recommendation="-",
    image_name="-",
):
    status_ar = STATUS_AR.get(
        str(status),
        str(status),
    )

    disease_missing = (
        disease_name is None
        or pd.isna(disease_name)
        or not str(disease_name).strip()
        or str(disease_name).strip().lower() in {
            "none",
            "nan",
            "-"
        }
    )

    disease_ar = (
        (
            "فراولة سليمة"
            if str(status) == "Healthy"
            else "-"
        )
        if disease_missing
        else disease_name_ar(
            disease_name
        )
    )

    html = f"""
    <!doctype html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="utf-8">
        <title>{escape(str(title))}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                direction: rtl;
                margin: 40px;
                color: #25364d;
                background: #ffffff;
            }}
            .header {{
                border-bottom: 4px solid #e93672;
                padding-bottom: 14px;
                margin-bottom: 24px;
            }}
            h1 {{
                color: #c91f59;
                margin: 0 0 8px 0;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
                margin: 20px 0;
            }}
            .card {{
                border: 1px solid #ecd6df;
                border-radius: 14px;
                padding: 14px;
                background: #fff8fb;
            }}
            .label {{
                color: #7a6971;
                font-size: 13px;
                margin-bottom: 6px;
            }}
            .value {{
                font-size: 19px;
                font-weight: 700;
            }}
            .recommendation {{
                margin-top: 20px;
                border: 1px solid #efd6ad;
                border-radius: 14px;
                padding: 16px;
                background: #fffaf0;
                line-height: 1.8;
            }}
            .note {{
                margin-top: 28px;
                color: #777;
                font-size: 12px;
                line-height: 1.7;
            }}
            @media print {{
                button {{
                    display: none;
                }}
                body {{
                    margin: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>عين المزارع - تقرير فحص</h1>
            <div>Farmer Eye AI</div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="label">الحالة</div>
                <div class="value">{escape(status_ar)}</div>
            </div>

            <div class="card">
                <div class="label">التشخيص</div>
                <div class="value">{escape(str(disease_ar))}</div>
            </div>

            <div class="card">
                <div class="label">درجة الثقة</div>
                <div class="value">{float(confidence) * 100:.2f}%</div>
            </div>

            <div class="card">
                <div class="label">المناطق المكتشفة</div>
                <div class="value">{int(detected_objects or 0)}</div>
            </div>

            <div class="card">
                <div class="label">تاريخ الفحص</div>
                <div class="value">{escape(str(analysis_date))}</div>
            </div>

            <div class="card">
                <div class="label">اسم الصورة</div>
                <div class="value">{escape(str(image_name))}</div>
            </div>
        </div>

        <div class="recommendation">
            <strong>الإرشاد المقترح</strong><br>
            {escape(str(recommendation))}
        </div>

        <div class="note">
            هذا التقرير أداة مساعدة لاتخاذ قرار أولي ولا يغني عن الفحص الميداني
            بواسطة مهندس زراعي عند وجود إصابة شديدة أو أعراض غير واضحة.
        </div>
    </body>
    </html>
    """

    return html


def extract_zip_images(
    uploaded_zip
):
    extract_root = (
        TEMP_DIR /
        f"zip_{time.time_ns()}"
    )

    extract_root.mkdir(
        parents=True,
        exist_ok=True
    )

    supported = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = []

    with zipfile.ZipFile(
        io.BytesIO(
            uploaded_zip.getvalue()
        )
    ) as archive:

        for member in archive.infolist():

            if member.is_dir():
                continue

            member_path = Path(
                member.filename
            )

            if (
                member_path.suffix.lower()
                not in supported
            ):
                continue

            safe_name = (
                f"{time.time_ns()}_"
                f"{member_path.name}"
            )

            target = (
                extract_root /
                safe_name
            )

            with archive.open(
                member
            ) as source, open(
                target,
                "wb"
            ) as destination:

                destination.write(
                    source.read()
                )

            image_paths.append(
                target
            )

    return image_paths


def build_gradcam(result):
    if result["status"] == "Uncertain":
        return None

    detections = result["detections"]

    if not detections:
        return None

    disease_detections = [
        detection
        for detection in detections
        if detection["class_name"] != "Healthy Strawberry"
    ]

    target = max(
        disease_detections
        if disease_detections
        else detections,
        key=lambda item: item["confidence"],
    )

    target_class_id = int(
        target["class_id"]
    )

    image_path = Path(
        result["original_image_path"]
    )

    prediction = prediction_model.predict(
        source=str(image_path),
        conf=0.50,
        iou=0.70,
        imgsz=640,
        device=(
            0
            if torch.cuda.is_available()
            else "cpu"
        ),
        retina_masks=True,
        verbose=False,
    )[0]

    target_mask = None

    if (
        prediction.boxes is not None
        and
        prediction.masks is not None
    ):

        class_ids = (
            prediction.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        masks = (
            prediction.masks.data
            .detach()
            .cpu()
            .numpy()
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is not None:

            height, width = (
                image.shape[:2]
            )

            indexes = np.where(
                class_ids
                ==
                target_class_id
            )[0]

            if len(indexes):

                target_mask = np.zeros(
                    (
                        height,
                        width
                    ),
                    dtype=np.float32,
                )

                for index in indexes:

                    mask = cv2.resize(
                        masks[index],
                        (
                            width,
                            height
                        ),
                    )

                    target_mask = np.maximum(
                        target_mask,
                        mask,
                    )

    return generate_gradcam(
        image_path=image_path,
        target_class_id=target_class_id,
        target_mask=target_mask,
        prediction_id=result["prediction_id"],
    )


def show_result(result, expert_mode=False):
    status = result["status"]
    status_ar = STATUS_AR.get(
        status,
        status
    )

    disease_ar = disease_name_ar(
        result["disease_name"]
    )

    recommendation = recommendation_for(
        result
    )

    confidence_text = confidence_label(
        result["confidence"]
    )

    disease_label = (
        disease_ar
        if status == "Diseased"
        else (
            "فراولة سليمة"
            if status == "Healthy"
            else "النتيجة غير مؤكدة"
        )
    )

    status_class = (
        "status-good"
        if status == "Healthy"
        else "status-bad"
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="{status_class}">{status_ar}</div>
            <div class="result-title">{disease_label}</div>
            <span class="confidence-pill">
                {confidence_text} • {result['confidence_percentage']:.2f}%
            </span>
            <div class="action-card">
                <strong>ماذا أفعل الآن؟</strong><br>
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    cols = st.columns(3)

    cols[0].metric(
        "موثوقية التشخيص",
        f"{result['confidence_percentage']:.2f}%",
    )

    cols[1].metric(
        "مناطق الإصابة",
        result["detected_objects"],
    )

    cols[2].metric(
        "مستوى المتابعة",
        RISK_AR.get(status, "-"),
    )

    gradcam_result = None

    if status != "Uncertain":

        try:
            with st.spinner(
                "جاري تجهيز التفسير البصري..."
            ):
                gradcam_result = build_gradcam(
                    result
                )

        except Exception as error:

            if expert_mode:
                st.warning(
                    f"تعذر إنشاء Grad-CAM: {error}"
                )
            else:
                st.info(
                    "التفسير البصري غير متاح لهذه الصورة."
                )

    st.markdown("### صور الفحص")

    if expert_mode:
        tab_labels = [
            "الصورة الأصلية",
            "التقسيم والتشخيص",
            "Grad-CAM",
        ]
    else:
        tab_labels = [
            "الصورة الأصلية",
            "أماكن الإصابة",
            "لماذا أعطى النظام هذه النتيجة؟",
        ]

    tab1, tab2, tab3 = st.tabs(
        tab_labels
    )

    with tab1:
        st.image(
            result["original_image_path"],
            use_container_width=True,
        )

    with tab2:
        st.image(
            result["prediction_image_path"],
            use_container_width=True,
        )

        if not expert_mode:
            st.caption(
                "الحدود والألوان توضح المناطق التي حددها النظام داخل الصورة."
            )

    with tab3:
        if gradcam_result:

            st.image(
                gradcam_result["gradcam_path"],
                use_container_width=True,
            )

            if expert_mode:
                st.caption(
                    "Grad-CAM يوضح المناطق الأكثر تأثيرًا في قرار النموذج."
                )
            else:
                st.caption(
                    "المناطق المضيئة هي الأجزاء التي ركز عليها النظام أثناء التشخيص."
                )

        else:
            st.info(
                "لا يوجد تفسير بصري متاح لهذه النتيجة."
            )

    if expert_mode:
        report_html = build_printable_report_html(
            title="Farmer Eye AI Report",
            status=result["status"],
            disease_name=result["disease_name"],
            confidence=result["confidence"],
            detected_objects=result["detected_objects"],
            analysis_date="الفحص الحالي",
            recommendation=recommendation,
            image_name=result["image_name"],
        )

        st.download_button(
            "تحميل تقرير الفحص للطباعة",
            data=report_html.encode("utf-8"),
            file_name=(
                f"farmer_eye_report_"
                f"{result['prediction_id'] or 'current'}.html"
            ),
            mime="text/html",
            use_container_width=True,
        )

        with st.expander(
            "التفاصيل التقنية",
            expanded=False,
        ):

            st.write(
                f"Class: {main_disease_name(result) or '-'}"
            )

            st.write(
                f"Confidence: {result['confidence_percentage']:.2f}%"
            )

            st.write(
                f"Detected Objects: {result['detected_objects']}"
            )

            st.write(
                f"Analysis Time: {result['analysis_time_ms']:.2f} ms"
            )

            detections = pd.DataFrame(
                result["detections"]
            )

            if not detections.empty:
                st.dataframe(
                    detections,
                    use_container_width=True,
                    hide_index=True,
                )

    prediction_id = result["prediction_id"]

    if prediction_id is not None:

        st.markdown(
            "#### هل كان التشخيص مفيدًا؟"
        )

        feedback_cols = st.columns(2)

        with feedback_cols[0]:
            if st.button(
                "نعم",
                key=f"correct_{prediction_id}",
            ):
                update_prediction_feedback(
                    prediction_id,
                    "Correct",
                )
                st.success(
                    "تم تسجيل تقييمك."
                )

        with feedback_cols[1]:
            if st.button(
                "لا",
                key=f"incorrect_{prediction_id}",
            ):
                update_prediction_feedback(
                    prediction_id,
                    "Incorrect",
                )
                st.success(
                    "تم تسجيل تقييمك."
                )


def render_history_details(row, expert_mode=False):
    status = str(row.get("status", ""))
    disease = row.get("disease_name")

    status_ar = STATUS_AR.get(
        status,
        status
    )

    confidence = float(
        row.get("confidence", 0)
        or 0
    )

    st.markdown(
        f"""
        <div class="card">
            <span class="mini-badge">{status_ar}</span>
            <h3 style="margin:12px 0 6px">
                {disease_name_ar(disease) if disease else "فراولة سليمة"}
            </h3>
            <div class="subtle">
                {row.get("analysis_date", "-")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "الثقة",
        f"{confidence * 100:.2f}%",
    )

    c2.metric(
        "المناطق",
        int(
            row.get(
                "detected_objects",
                0
            )
            or 0
        ),
    )

    feedback_value = row.get(
        "user_feedback"
    )

    feedback_text = (
        "صحيحة"
        if feedback_value == "Correct"
        else (
            "غير صحيحة"
            if feedback_value == "Incorrect"
            else "لم يتم التقييم"
        )
    )

    c3.metric(
        "تقييم المستخدم",
        feedback_text,
    )

    original_path = existing_path(
        row.get("original_image")
    )

    segmentation_path = existing_path(
        row.get("segmentation_image")
    )

    gradcam_path = existing_path(
        row.get("gradcam_image")
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "الصورة الأصلية",
            "أماكن الإصابة",
            (
                "Grad-CAM"
                if expert_mode
                else "تفسير التشخيص"
            ),
        ]
    )

    with tab1:
        if original_path:
            st.image(
                str(original_path),
                use_container_width=True,
            )
        else:
            st.info(
                "الصورة الأصلية غير متاحة في التخزين الحالي."
            )

    with tab2:
        if segmentation_path:
            st.image(
                str(segmentation_path),
                use_container_width=True,
            )
        else:
            st.info(
                "صورة تحديد الإصابة غير متاحة."
            )

    with tab3:
        if gradcam_path:
            st.image(
                str(gradcam_path),
                use_container_width=True,
            )
        else:
            st.info(
                "صورة تفسير التشخيص غير متاحة لهذا السجل."
            )


    if expert_mode:

        row_recommendation = RECOMMENDATIONS_AR.get(
            str(
                row.get(
                    "disease_name",
                    ""
                )
                or
                "Healthy Strawberry"
            ).split(",")[0].strip(),
            "راجع الحالة ميدانيًا إذا استمرت الأعراض أو توسعت الإصابة.",
        )

        report_html = build_printable_report_html(
            title="Farmer Eye AI Historical Report",
            status=status,
            disease_name=disease,
            confidence=confidence,
            detected_objects=(
                row.get(
                    "detected_objects",
                    0
                )
                or
                0
            ),
            analysis_date=row.get(
                "analysis_date",
                "-"
            ),
            recommendation=row_recommendation,
            image_name=row.get(
                "image_name",
                "-"
            ),
        )

        st.download_button(
            "تحميل التقرير للطباعة",
            data=report_html.encode("utf-8"),
            file_name=(
                f"farmer_eye_history_"
                f"{int(row.get('id', 0))}.html"
            ),
            mime="text/html",
            key=(
                f"history_report_"
                f"{int(row.get('id', 0))}"
            ),
            use_container_width=True,
        )


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">
            <div class="logo">🍓</div>
            <h2>عين المزارع</h2>
            <p>Farmer Eye AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user_mode = st.radio(
        "طريقة العرض",
        [
            "مزارع",
            "مهندس زراعي",
        ],
        horizontal=True,
    )

    expert_mode = (
        user_mode
        ==
        "مهندس زراعي"
    )

    st.write("")

    page = st.radio(
        "القائمة",
        [
            "الرئيسية",
            "فحص صورة",
            "فحص مجموعة صور",
            "التحليلات",
            "السجل",
            "عن النظام",
        ],
    )

    st.write("")

    st.markdown(
        """
        <div class="card" style="padding:16px">
            <strong>حالة الخدمة</strong><br><br>
            <span class="status-good">● جاهز للاستخدام</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Main App
# ==============================================================================

render_hero()


if page == "الرئيسية":

    st.markdown(
        """
        <div class="section-title">
            <h3>كيف تستخدم النظام؟</h3>
            <span class="subtle">
                ثلاث خطوات فقط للحصول على نتيجة واضحة
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

    steps = [
        (
            "1",
            "ارفع الصورة",
            "اختر صورة واضحة للثمرة أو الورقة، ويفضل أن تكون الإضاءة جيدة.",
        ),
        (
            "2",
            "ابدأ الفحص",
            "يقوم النظام بتحليل الصورة وتحديد الحالة ومكان الإصابة.",
        ),
        (
            "3",
            "راجع الإرشاد",
            "شاهد النتيجة واقرأ الخطوة المقترحة، ويمكن للمهندس فتح التفاصيل الفنية.",
        ),
    ]

    for column, (
        number,
        title,
        description
    ) in zip(
        columns,
        steps
    ):

        with column:

            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{number}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    statistics = get_general_statistics()

    st.markdown(
        "### نظرة سريعة"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "إجمالي الفحوصات",
        statistics["total_images"],
    )

    c2.metric(
        "حالات سليمة",
        statistics["healthy_images"],
    )

    c3.metric(
        "حالات مصابة",
        statistics["diseased_images"],
    )

    c4.metric(
        "غير مؤكدة",
        statistics["uncertain_images"],
    )

    st.markdown(
        """
        <div class="card" style="margin-top:18px">
            <h3 style="margin-top:0">🍓 ابدأ بفحص صورة جديدة</h3>
            <p class="subtle">
                افتح صفحة <strong>فحص صورة</strong>،
                ارفع الصورة واضغط زر بدء الفحص.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "فحص صورة":

    st.markdown(
        """
        <div class="section-title">
            <h3>فحص صورة فراولة</h3>
            <span class="subtle">
                ارفع صورة أو التقط صورة مباشرة بالكاميرا
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_tab1, source_tab2 = st.tabs(
        [
            "رفع صورة",
            "التقاط بالكاميرا",
        ]
    )

    uploaded_file = None

    with source_tab1:

        uploaded_file = st.file_uploader(
            "اختر صورة",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
                "webp",
            ],
            label_visibility="collapsed",
        )

    with source_tab2:

        camera_file = st.camera_input(
            "التقط صورة واضحة للثمرة أو الورقة"
        )

        if camera_file is not None:
            uploaded_file = camera_file

    upload_col, tips_col = st.columns(
        [1.25, .75]
    )

    with upload_col:

        if uploaded_file:

            st.image(
                uploaded_file,
                caption="الصورة المختارة",
                width=460,
            )

            if st.button(
                "ابدأ الفحص",
                use_container_width=True,
            ):

                image_path = save_upload(
                    uploaded_file
                )

                with st.spinner(
                    "جاري فحص الصورة..."
                ):

                    result = analyze_image(
                        image_path,
                        save_to_database=True,
                    )

                st.session_state[
                    "single_result"
                ] = result

        else:

            st.info(
                "اختر صورة من الجهاز أو استخدم الكاميرا."
            )

    with tips_col:

        st.markdown(
            """
            <div class="card">
                <h3 style="margin-top:0">نصائح لصورة أفضل</h3>
                <p class="subtle">
                    • استخدم إضاءة جيدة<br><br>
                    • قرب الكاميرا من الثمرة أو الورقة<br><br>
                    • تجنب الصور المهزوزة<br><br>
                    • اجعل الجزء المصاب ظاهرًا بوضوح
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if (
        "single_result"
        in st.session_state
    ):

        st.write("")

        show_result(
            st.session_state[
                "single_result"
            ],
            expert_mode=expert_mode,
        )


elif page == "فحص مجموعة صور":

    st.markdown(
        """
        <div class="section-title">
            <h3>فحص مجموعة صور</h3>
            <span class="subtle">
                ارفع صور متعددة أو مجلد صور كامل
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    batch_tab1, batch_tab2, batch_tab3 = st.tabs(
        [
            "اختيار صور متعددة",
            "اختيار مجلد كامل",
            "رفع مجلد ZIP",
        ]
    )

    selected_files = []
    selected_paths = []

    with batch_tab1:

        manual_files = st.file_uploader(
            "اختر عدة صور",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
                "webp",
            ],
            accept_multiple_files=True,
            key="batch_manual",
        )

        if manual_files:
            selected_files.extend(
                manual_files
            )

    with batch_tab2:

        try:
            folder_files = st.file_uploader(
                "اختر مجلد الصور",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "bmp",
                    "webp",
                ],
                accept_multiple_files="directory",
                key="batch_directory",
            )

            if folder_files:
                selected_files.extend(
                    folder_files
                )

        except Exception:

            st.info(
                "اختيار المجلد غير مدعوم في هذا المتصفح. "
                "استخدم تبويب ZIP بدلًا منه."
            )

    with batch_tab3:

        zip_file = st.file_uploader(
            "ارفع ملف ZIP يحتوي على الصور",
            type=[
                "zip",
            ],
            key="batch_zip",
        )

        if zip_file is not None:

            try:
                selected_paths = extract_zip_images(
                    zip_file
                )

                st.success(
                    f"تم العثور على {len(selected_paths)} صورة داخل الملف."
                )

            except zipfile.BadZipFile:

                st.error(
                    "ملف ZIP غير صالح."
                )

    total_selected = (
        len(
            selected_files
        )
        +
        len(
            selected_paths
        )
    )

    if total_selected:

        st.info(
            f"إجمالي الصور الجاهزة للفحص: {total_selected}"
        )

    if (
        total_selected
        and
        st.button(
            "ابدأ فحص المجموعة",
            use_container_width=True,
        )
    ):

        results = []

        progress = st.progress(
            0
        )

        all_items = []

        for uploaded_file in selected_files:
            all_items.append(
                (
                    "upload",
                    uploaded_file
                )
            )

        for image_path in selected_paths:
            all_items.append(
                (
                    "path",
                    image_path
                )
            )

        for index, (
            source_type,
            source_value
        ) in enumerate(
            all_items
        ):

            if source_type == "upload":

                image_path = save_upload(
                    source_value
                )

            else:

                image_path = Path(
                    source_value
                )

            result = analyze_image(
                image_path,
                save_to_database=True,
            )

            result[
                "batch_gradcam_path"
            ] = None

            if (
                result[
                    "status"
                ]
                !=
                "Uncertain"
            ):

                try:

                    gradcam_result = build_gradcam(
                        result
                    )

                    if gradcam_result:

                        result[
                            "batch_gradcam_path"
                        ] = gradcam_result[
                            "gradcam_path"
                        ]

                except Exception:

                    result[
                        "batch_gradcam_path"
                    ] = None

            results.append(
                result
            )

            progress.progress(
                (
                    index + 1
                )
                /
                len(
                    all_items
                )
            )

        st.session_state[
            "batch_results"
        ] = results

    if (
        "batch_results"
        in st.session_state
    ):

        results = st.session_state[
            "batch_results"
        ]

        st.markdown(
            "### نتائج المجموعة"
        )

        for index, result in enumerate(
            results
        ):

            status_text = STATUS_AR.get(
                result[
                    "status"
                ],
                result[
                    "status"
                ],
            )

            disease_text = disease_name_ar(
                result[
                    "disease_name"
                ]
            )

            with st.expander(
                (
                    f"{index + 1}. "
                    f"{status_text}"
                    f" - "
                    f"{disease_text}"
                    f" - "
                    f"{result['confidence_percentage']:.1f}%"
                ),
                expanded=(
                    index == 0
                ),
            ):

                info_col, image_col = st.columns(
                    [.8, 1.2]
                )

                with info_col:

                    st.metric(
                        "درجة الثقة",
                        f"{result['confidence_percentage']:.2f}%",
                    )

                    st.write(
                        f"**الحالة:** {status_text}"
                    )

                    st.write(
                        f"**المرض:** {disease_text}"
                    )

                    st.write(
                        f"**المناطق المكتشفة:** "
                        f"{result['detected_objects']}"
                    )

                    st.write(
                        f"**الإرشاد:** "
                        f"{recommendation_for(result)}"
                    )

                with image_col:

                    st.image(
                        result[
                            "original_image_path"
                        ],
                        width=420,
                    )

                result_tabs = st.tabs(
                    [
                        "الصورة الأصلية",
                        "أماكن الإصابة",
                        (
                            "Grad-CAM"
                            if expert_mode
                            else
                            "تفسير التشخيص"
                        ),
                    ]
                )

                with result_tabs[0]:

                    st.image(
                        result[
                            "original_image_path"
                        ],
                        width=520,
                    )

                with result_tabs[1]:

                    st.image(
                        result[
                            "prediction_image_path"
                        ],
                        width=520,
                    )

                with result_tabs[2]:

                    batch_gradcam_path = result.get(
                        "batch_gradcam_path"
                    )

                    if (
                        batch_gradcam_path
                        and
                        Path(
                            batch_gradcam_path
                        ).exists()
                    ):

                        st.image(
                            batch_gradcam_path,
                            width=520,
                        )

                        if not expert_mode:

                            st.caption(
                                "المناطق المضيئة هي الأجزاء التي ركز عليها النظام أثناء التشخيص."
                            )

                    else:

                        st.info(
                            "التفسير البصري غير متاح لهذه الصورة."
                        )

                if expert_mode:

                    report_html = build_printable_report_html(
                        title="Farmer Eye AI Batch Report",
                        status=result["status"],
                        disease_name=result["disease_name"],
                        confidence=result["confidence"],
                        detected_objects=result["detected_objects"],
                        analysis_date="فحص مجموعة",
                        recommendation=recommendation_for(result),
                        image_name=result["image_name"],
                    )

                    st.download_button(
                        "تحميل تقرير هذه الصورة للطباعة",
                        data=report_html.encode("utf-8"),
                        file_name=(
                            f"farmer_eye_report_"
                            f"{result['prediction_id']}.html"
                        ),
                        mime="text/html",
                        key=(
                            f"batch_report_"
                            f"{result['prediction_id']}"
                        ),
                        use_container_width=True,
                    )


elif page == "التحليلات":

    st.markdown(
        """
        <div class="section-title">
            <h3>لوحة التحليلات</h3>
            <span class="subtle">
                ملخص ذكي للحالات المسجلة
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = get_prediction_history()

    statistics = get_general_statistics()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "إجمالي الفحوصات",
        statistics["total_images"],
    )

    k2.metric(
        "سليمة",
        statistics["healthy_images"],
    )

    k3.metric(
        "مصابة",
        statistics["diseased_images"],
    )

    k4.metric(
        "متوسط الثقة",
        f"{statistics['average_confidence'] * 100:.1f}%",
    )

    st.write("")

    if history.empty:

        st.info(
            "لا توجد بيانات كافية لإنشاء التحليلات."
        )

    else:

        history_view = history.copy()

        history_view[
            "analysis_date"
        ] = pd.to_datetime(
            history_view[
                "analysis_date"
            ],
            errors="coerce",
        )

        history_view[
            "day"
        ] = (
            history_view[
                "analysis_date"
            ]
            .dt.date
            .astype(str)
        )

        status_counts = (
            history_view[
                "status"
            ]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "status",
            "count",
        ]

        status_counts[
            "الحالة"
        ] = status_counts[
            "status"
        ].map(
            STATUS_AR
        )

        disease_data = history_view[
            (
                history_view[
                    "status"
                ]
                ==
                "Diseased"
            )
            &
            history_view[
                "disease_name"
            ].notna()
        ].copy()

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            fig_status = px.pie(
                status_counts,
                names="الحالة",
                values="count",
                hole=.55,
                title="توزيع الحالات",
            )

            fig_status.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10,
                ),
                legend_title_text="",
            )

            st.plotly_chart(
                fig_status,
                use_container_width=True,
            )

        with chart_col2:

            if not disease_data.empty:

                disease_counts = (
                    disease_data[
                        "disease_name"
                    ]
                    .value_counts()
                    .reset_index()
                )

                disease_counts.columns = [
                    "disease",
                    "count",
                ]

                disease_counts[
                    "المرض"
                ] = disease_counts[
                    "disease"
                ].apply(
                    disease_name_ar
                )

                fig_disease = px.bar(
                    disease_counts,
                    x="count",
                    y="المرض",
                    orientation="h",
                    title="أكثر الأمراض ظهورًا",
                    labels={
                        "count":
                            "عدد الحالات",
                    },
                )

                fig_disease.update_layout(
                    margin=dict(
                        l=10,
                        r=10,
                        t=55,
                        b=10,
                    )
                )

                st.plotly_chart(
                    fig_disease,
                    use_container_width=True,
                )

            else:

                st.info(
                    "لا توجد حالات مرضية كافية لعرض توزيع الأمراض."
                )

        daily_counts = (
            history_view
            .groupby(
                "day"
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        fig_daily = px.line(
            daily_counts,
            x="day",
            y="count",
            markers=True,
            title="عدد الفحوصات بمرور الوقت",
            labels={
                "day":
                    "التاريخ",
                "count":
                    "عدد الفحوصات",
            },
        )

        st.plotly_chart(
            fig_daily,
            use_container_width=True,
        )

        if not disease_data.empty:

            disease_conf = (
                disease_data
                .groupby(
                    "disease_name",
                    as_index=False
                )[
                    "confidence"
                ]
                .mean()
            )

            disease_conf[
                "المرض"
            ] = disease_conf[
                "disease_name"
            ].apply(
                disease_name_ar
            )

            disease_conf[
                "متوسط الثقة %"
            ] = (
                disease_conf[
                    "confidence"
                ]
                *
                100
            ).round(
                2
            )

            fig_conf = px.bar(
                disease_conf,
                x="المرض",
                y="متوسط الثقة %",
                title="متوسط الثقة لكل مرض",
            )

            st.plotly_chart(
                fig_conf,
                use_container_width=True,
            )


elif page == "السجل":

    st.markdown(
        """
        <div class="section-title">
            <h3>سجل الفحوصات</h3>
            <span class="subtle">
                ابحث وافتح أي فحص سابق بالتفصيل
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = get_prediction_history()

    if history.empty:

        st.info(
            "لا يوجد سجل فحوصات حتى الآن."
        )

    else:

        filter_col1, filter_col2 = st.columns(
            2
        )

        with filter_col1:

            status_filter = st.selectbox(
                "الحالة",
                [
                    "الكل",
                    "سليمة",
                    "مصابة",
                    "غير مؤكدة",
                ],
            )

        with filter_col2:

            disease_options = [
                value
                for value
                in history[
                    "disease_name"
                ]
                .dropna()
                .unique()
                .tolist()
                if str(value).strip()
            ]

            disease_filter = st.selectbox(
                "المرض",
                [
                    "الكل",
                ]
                +
                [
                    disease_name_ar(
                        value
                    )
                    for value
                    in disease_options
                ],
            )

        filtered = history.copy()

        if status_filter != "الكل":

            reverse_status = {
                value: key
                for key, value
                in STATUS_AR.items()
            }

            filtered = filtered[
                filtered[
                    "status"
                ]
                ==
                reverse_status[
                    status_filter
                ]
            ]

        if disease_filter != "الكل":

            selected_index = [
                disease_name_ar(
                    value
                )
                for value
                in disease_options
            ].index(
                disease_filter
            )

            selected_disease = disease_options[
                selected_index
            ]

            filtered = filtered[
                filtered[
                    "disease_name"
                ]
                ==
                selected_disease
            ]

        if filtered.empty:

            st.info(
                "لا توجد نتائج مطابقة للفلاتر الحالية."
            )

        else:

            select_options = {}

            for _, row in filtered.iterrows():

                label = (
                    f"فحص #{int(row['id'])} | "
                    f"{STATUS_AR.get(str(row['status']), str(row['status']))} | "
                    f"{disease_name_ar(row['disease_name']) if pd.notna(row['disease_name']) else 'فراولة سليمة'} | "
                    f"{str(row['analysis_date'])}"
                )

                select_options[
                    label
                ] = int(
                    row[
                        "id"
                    ]
                )

            selected_label = st.selectbox(
                "اختر الفحص لعرض التفاصيل",
                list(
                    select_options.keys()
                ),
            )

            selected_id = select_options[
                selected_label
            ]

            selected_row = filtered[
                filtered[
                    "id"
                ]
                ==
                selected_id
            ].iloc[
                0
            ]

            render_history_details(
                selected_row,
                expert_mode=expert_mode,
            )

            with st.expander(
                "عرض الجدول المختصر",
                expanded=False,
            ):

                table_view = filtered[
                    [
                        "analysis_date",
                        "status",
                        "disease_name",
                        "confidence",
                        "detected_objects",
                        "user_feedback",
                    ]
                ].copy()

                table_view[
                    "status"
                ] = (
                    table_view[
                        "status"
                    ]
                    .map(
                        STATUS_AR
                    )
                    .fillna(
                        table_view[
                            "status"
                        ]
                    )
                )

                table_view[
                    "disease_name"
                ] = (
                    table_view[
                        "disease_name"
                    ]
                    .fillna(
                        "-"
                    )
                    .apply(
                        disease_name_ar
                    )
                )

                table_view[
                    "confidence"
                ] = (
                    table_view[
                        "confidence"
                    ]
                    .fillna(
                        0
                    )
                    .mul(
                        100
                    )
                    .round(
                        2
                    )
                )

                table_view = table_view.rename(
                    columns={
                        "analysis_date":
                            "التاريخ",
                        "status":
                            "الحالة",
                        "disease_name":
                            "المرض",
                        "confidence":
                            "الثقة %",
                        "detected_objects":
                            "المناطق",
                        "user_feedback":
                            "التقييم",
                    }
                )

                st.dataframe(
                    table_view,
                    use_container_width=True,
                    hide_index=True,
                )


elif page == "عن النظام":

    st.markdown(
        """
        <div class="card">
            <h2 style="margin-top:0;color:#c91f59">
                عن عين المزارع
            </h2>
            <p>
                نظام ذكي للمساعدة في الكشف المبكر عن أمراض الفراولة
                من خلال تحليل الصور وتحديد مواضع الإصابة.
            </p>
            <p class="subtle">
                النتائج مساعدة لاتخاذ قرار أولي، ولا تغني عن الفحص الميداني
                عند وجود إصابة شديدة أو أعراض غير واضحة.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if expert_mode:

        st.write("")

        with st.expander(
            "التفاصيل التقنية",
            expanded=False,
        ):

            st.write(
                "Model: YOLO11n-seg"
            )

            st.write(
                "Input Size: 640 × 640"
            )

            st.write(
                "Confidence Threshold: 50%"
            )

            st.write(
                "Explainability: Multi-Layer Grad-CAM"
            )

            st.write(
                f"Classes: {len(MODEL_NAMES)}"
            )

            class_table = pd.DataFrame(
                [
                    {
                        "Class":
                            class_name,
                        "Arabic":
                            DISEASE_AR.get(
                                class_name,
                                class_name,
                            ),
                    }
                    for class_name
                    in MODEL_NAMES.values()
                ]
            )

            st.dataframe(
                class_table,
                use_container_width=True,
                hide_index=True,
            )


st.markdown(
    """
    <div class="footer">
        🍓 عين المزارع • Farmer Eye AI
    </div>
    """,
    unsafe_allow_html=True,
)
