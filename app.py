
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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


st.set_page_config(
    page_title="عين المزارع | Farmer Eye AI",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    "Angular Leafspot": "اعزل الأوراق المصابة، قلل بلل الأوراق، وراقب انتشار الأعراض.",
    "Anthracnose Fruit Rot": "أزل الثمار المصابة، تجنب ملامسة الثمار للتربة، وحسن التهوية.",
    "Blossom Blight": "تخلص من الأزهار المصابة وحافظ على تهوية جيدة حول النبات.",
    "Gray Mold": "أزل الأجزاء المصابة، قلل الرطوبة، وتجنب الري فوق المجموع الخضري.",
    "Leaf Spot": "أزل الأوراق شديدة الإصابة، حسن التهوية، وراقب البقع الجديدة.",
    "Powdery Mildew Fruit": "افصل الثمار المصابة وقلل الرطوبة الزائدة مع متابعة باقي الثمار.",
    "Powdery Mildew Leaf": "أزل الأوراق شديدة الإصابة وحسن حركة الهواء حول النباتات.",
    "Healthy Strawberry": "لا توجد علامات مرضية واضحة. استمر في المتابعة الدورية.",
}


st.markdown(
    """
    <style>
    :root {
        --pink:#ef3f78;
        --pink-dark:#d92f68;
        --pink-soft:#fff2f7;
        --pink-border:#f4c6d6;
        --green:#2ca96b;
        --green-soft:#edf9f3;
        --navy:#25364d;
        --text:#263238;
        --muted:#667085;
        --card:#ffffff;
        --shadow:0 12px 30px rgba(42, 22, 33, .08);
    }

    html, body, [class*="css"] {
        direction: rtl;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, #ffe4ee 0, transparent 22%),
            linear-gradient(180deg, #fff9fb 0%, #fff5f8 55%, #ffffff 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #fff6f9 100%);
        border-left: 1px solid #f2dce4;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    .side-brand {
        text-align:center;
        background:white;
        border:1px solid #f1d4df;
        box-shadow:var(--shadow);
        border-radius:24px;
        padding:20px 14px;
        margin-bottom:16px;
    }

    .berry-logo {
        font-size:54px;
        line-height:1;
        margin-bottom:6px;
    }

    .side-brand h2 {
        margin:0;
        color:var(--pink-dark);
        font-size:1.55rem;
        font-weight:800;
    }

    .side-brand p {
        margin:7px 0 0 0;
        color:var(--muted);
        font-size:.85rem;
    }

    .hero {
        position:relative;
        overflow:hidden;
        border-radius:28px;
        padding:34px 38px;
        margin-bottom:24px;
        color:white;
        box-shadow:0 18px 45px rgba(197, 34, 91, .22);
        background:
            radial-gradient(circle at 12% 25%, rgba(255,255,255,.18), transparent 17%),
            radial-gradient(circle at 88% 15%, rgba(255,255,255,.16), transparent 19%),
            linear-gradient(135deg, #ee2f6f 0%, #f65485 48%, #cf245c 100%);
    }

    .hero:before,
    .hero:after {
        content:"🍓";
        position:absolute;
        font-size:88px;
        opacity:.18;
        filter:saturate(1.1);
    }

    .hero:before {
        left:28px;
        top:8px;
        transform:rotate(-12deg);
    }

    .hero:after {
        right:34px;
        bottom:-18px;
        transform:rotate(14deg);
    }

    .hero h1 {
        margin:0;
        font-size:2.7rem;
        font-weight:900;
        letter-spacing:-1px;
    }

    .hero p {
        margin:10px 0 0 0;
        max-width:760px;
        font-size:1.05rem;
        opacity:.96;
        line-height:1.9;
    }

    .hero-badge {
        display:inline-block;
        margin-top:16px;
        background:rgba(255,255,255,.18);
        border:1px solid rgba(255,255,255,.3);
        padding:8px 14px;
        border-radius:999px;
        font-weight:700;
        backdrop-filter:blur(5px);
    }

    .section-title {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        margin:8px 0 14px 0;
    }

    .section-title h3 {
        margin:0;
        font-size:1.25rem;
        font-weight:800;
        color:#2c3340;
    }

    .subtle {
        color:var(--muted);
        font-size:.92rem;
    }

    .soft-card,
    .result-card,
    .diagnosis-card,
    .info-card {
        background:var(--card);
        border:1px solid #f0d7e0;
        border-radius:22px;
        box-shadow:var(--shadow);
    }

    .soft-card {
        padding:20px;
        min-height:140px;
    }

    .step-card {
        background:white;
        border:1px solid #f1d9e2;
        border-radius:20px;
        padding:20px;
        box-shadow:0 9px 24px rgba(45, 27, 36, .05);
        min-height:150px;
    }

    .step-num {
        width:34px;
        height:34px;
        border-radius:50%;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        color:#fff;
        background:linear-gradient(135deg,var(--pink),var(--pink-dark));
        font-weight:800;
        margin-bottom:10px;
    }

    .step-card h4 {
        margin:4px 0 7px 0;
        color:#323844;
    }

    .step-card p {
        color:var(--muted);
        line-height:1.7;
        margin:0;
        font-size:.9rem;
    }

    .result-card {
        padding:24px;
        text-align:center;
        border-top:4px solid var(--pink);
    }

    .result-card h2 {
        margin:5px 0;
        font-size:2rem;
    }

    .diagnosis-name {
        color:var(--pink-dark);
        font-size:1.45rem;
        font-weight:900;
        margin:8px 0;
    }

    .confidence-pill {
        display:inline-block;
        margin-top:8px;
        padding:8px 14px;
        border-radius:999px;
        background:#fff0f5;
        color:var(--pink-dark);
        font-weight:800;
        border:1px solid #f7c9d9;
    }

    .recommendation {
        margin-top:14px;
        padding:14px 16px;
        border-radius:16px;
        background:linear-gradient(135deg,#fffaf1,#fff5e7);
        border:1px solid #f6ddb3;
        color:#6a4a18;
        line-height:1.8;
    }

    .status-good {
        color:var(--green);
        font-weight:800;
    }

    .status-bad {
        color:var(--pink-dark);
        font-weight:800;
    }

    div[data-testid="stMetric"] {
        background:white;
        border:1px solid #eed7df;
        border-radius:18px;
        padding:14px 16px;
        box-shadow:0 8px 20px rgba(40, 25, 34, .05);
    }

    div[data-testid="stMetric"] label {
        color:#697386 !important;
    }

    div[data-testid="stMetricValue"] {
        color:#26364b;
        font-weight:800;
    }

    div[data-testid="stFileUploader"] {
        background:white;
        border:1px dashed #ef8eae;
        border-radius:22px;
        padding:8px;
    }

    .stButton > button {
        width:100%;
        min-height:48px;
        border-radius:14px;
        border:none;
        font-weight:800;
        color:white;
        background:linear-gradient(135deg,var(--pink),var(--pink-dark));
        box-shadow:0 8px 18px rgba(217,47,104,.2);
    }

    .stButton > button:hover {
        color:white;
        border:none;
        transform:translateY(-1px);
    }

    div[data-baseweb="tab-list"] {
        gap:8px;
    }

    button[data-baseweb="tab"] {
        border-radius:12px;
        padding:8px 14px;
    }

    .footer {
        margin-top:30px;
        padding-top:16px;
        border-top:1px solid #efdce4;
        text-align:center;
        color:#8b7480;
        font-size:.85rem;
    }

    @media (max-width: 900px) {
        .hero h1 {font-size:2rem;}
        .hero {padding:26px 22px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero():
    st.markdown(
        """
        <div class="hero">
            <h1>عين المزارع 🍓</h1>
            <p>
                منصة ذكية لتحليل صور الفراولة والكشف المبكر عن الأمراض
                باستخدام YOLO11n-seg مع تحديد مناطق الإصابة وGrad-CAM
                لتفسير قرار النموذج.
            </p>
            <span class="hero-badge">YOLO11n-seg • Instance Segmentation • Grad-CAM</span>
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
    if not name:
        return "-"
    diseases = [value.strip() for value in name.split(",") if value.strip()]
    return "، ".join(DISEASE_AR.get(disease, disease) for disease in diseases)


def main_disease_name(result):
    name = result.get("disease_name")
    if not name:
        return "Healthy Strawberry" if result.get("status") == "Healthy" else None
    return name.split(",")[0].strip()


def recommendation_for(result):
    disease = main_disease_name(result)
    if disease is None:
        return "أعد التصوير في إضاءة واضحة وبزاوية أقرب للثمرة أو الورقة."
    return RECOMMENDATIONS_AR.get(
        disease,
        "اعزل الأجزاء المشتبه بها وراقب تطور الأعراض، واستعن بمتخصص زراعي عند الحاجة.",
    )


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
        disease_detections if disease_detections else detections,
        key=lambda item: item["confidence"],
    )

    target_class_id = int(target["class_id"])
    image_path = Path(result["original_image_path"])

    prediction = prediction_model.predict(
        source=str(image_path),
        conf=0.50,
        iou=0.70,
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu",
        retina_masks=True,
        verbose=False,
    )[0]

    target_mask = None

    if prediction.boxes is not None and prediction.masks is not None:
        class_ids = (
            prediction.boxes.cls.detach().cpu().numpy().astype(int)
        )
        masks = prediction.masks.data.detach().cpu().numpy()

        image = cv2.imread(str(image_path))
        if image is not None:
            height, width = image.shape[:2]
            indexes = np.where(class_ids == target_class_id)[0]

            if len(indexes):
                target_mask = np.zeros(
                    (height, width),
                    dtype=np.float32,
                )

                for index in indexes:
                    mask = cv2.resize(
                        masks[index],
                        (width, height),
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


def show_result(result):
    status_ar = STATUS_AR.get(result["status"], result["status"])
    disease_ar = disease_name_ar(result["disease_name"])
    recommendation = recommendation_for(result)

    disease_label = (
        disease_ar
        if result["status"] == "Diseased"
        else (
            "فراولة سليمة"
            if result["status"] == "Healthy"
            else "النتيجة غير مؤكدة"
        )
    )

    status_class = (
        "status-good"
        if result["status"] == "Healthy"
        else "status-bad"
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="{status_class}">{status_ar}</div>
            <div class="diagnosis-name">{disease_label}</div>
            <span class="confidence-pill">
                درجة الثقة {result['confidence_percentage']:.2f}%
            </span>
            <div class="recommendation">
                <strong>التوصية:</strong> {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "درجة الثقة",
        f"{result['confidence_percentage']:.2f}%",
    )
    metric_cols[1].metric(
        "المناطق المكتشفة",
        result["detected_objects"],
    )
    metric_cols[2].metric(
        "زمن التحليل",
        f"{result['analysis_time_ms']:.0f} ms",
    )

    gradcam_result = None

    if result["status"] != "Uncertain":
        try:
            with st.spinner("جاري إنشاء Grad-CAM..."):
                gradcam_result = build_gradcam(result)
        except Exception as error:
            st.warning(f"تعذر إنشاء Grad-CAM لهذه الصورة: {error}")

    st.markdown("### نتيجة التحليل")

    tab1, tab2, tab3 = st.tabs(
        [
            "الصورة الأصلية",
            "التقسيم والتشخيص",
            "تفسير Grad-CAM",
        ]
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

    with tab3:
        if gradcam_result:
            st.image(
                gradcam_result["gradcam_path"],
                use_container_width=True,
            )
            st.caption(
                "الخريطة توضح المناطق الأكثر تأثيرًا في قرار النموذج."
            )
        else:
            st.info("Grad-CAM غير متاح لهذه النتيجة.")

    prediction_id = result["prediction_id"]

    if prediction_id is not None:
        st.markdown("#### هل كانت النتيجة صحيحة؟")
        feedback_cols = st.columns(2)

        with feedback_cols[0]:
            if st.button(
                "نعم، النتيجة صحيحة",
                key=f"correct_{prediction_id}",
            ):
                update_prediction_feedback(
                    prediction_id,
                    "Correct",
                )
                st.success("تم تسجيل تقييمك.")

        with feedback_cols[1]:
            if st.button(
                "لا، النتيجة غير صحيحة",
                key=f"incorrect_{prediction_id}",
            ):
                update_prediction_feedback(
                    prediction_id,
                    "Incorrect",
                )
                st.success("تم تسجيل تقييمك.")


with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="berry-logo">🍓</div>
            <h2>عين المزارع</h2>
            <p>Farmer Eye AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "القائمة",
        [
            "الرئيسية",
            "تحليل صورة",
            "تحليل مجموعة صور",
            "الإحصائيات",
            "السجل",
            "عن النظام",
        ],
    )

    st.write("")
    st.markdown(
        """
        <div class="soft-card" style="min-height:auto">
            <strong>حالة النظام</strong><br><br>
            <span class="status-good">● متصل وجاهز</span><br>
            <span class="subtle">YOLO11n-seg • 640px • Confidence 50%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


hero()


if page == "الرئيسية":
    st.markdown(
        """
        <div class="section-title">
            <h3>كيف يعمل النظام؟</h3>
            <span class="subtle">3 خطوات بسيطة للحصول على التشخيص</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    steps = [
        (
            "1",
            "ارفع الصورة",
            "اختر صورة واضحة للثمرة أو الورقة بصيغة JPG أو PNG أو WEBP.",
        ),
        (
            "2",
            "ابدأ التحليل",
            "يحدد النموذج المرض ومناطق الإصابة باستخدام Instance Segmentation.",
        ),
        (
            "3",
            "راجع النتيجة",
            "شاهد التشخيص ونسبة الثقة وGrad-CAM والتوصية المناسبة.",
        ),
    ]

    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-num">{num}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    stats = get_general_statistics()

    st.markdown("### نظرة سريعة")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي التحليلات", stats["total_images"])
    c2.metric("صور سليمة", stats["healthy_images"])
    c3.metric("صور مصابة", stats["diseased_images"])
    c4.metric("غير مؤكدة", stats["uncertain_images"])

    st.markdown(
        """
        <div class="soft-card" style="margin-top:18px">
            <h3 style="margin-top:0">🍓 جاهز لفحص صورة فراولة؟</h3>
            <p class="subtle">
                انتقل إلى صفحة <strong>تحليل صورة</strong>،
                ارفع الصورة ثم اضغط ابدأ التحليل.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "تحليل صورة":
    st.markdown(
        """
        <div class="section-title">
            <h3>تحليل صورة فراولة</h3>
            <span class="subtle">ارفع صورة واحدة للحصول على التشخيص والتفسير البصري</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([1.35, 0.65])

    with upload_col:
        uploaded_file = st.file_uploader(
            "اختر صورة فراولة",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            if st.button(
                "ابدأ التحليل",
                use_container_width=True,
            ):
                image_path = save_upload(uploaded_file)

                with st.spinner(
                    "جاري تحليل الصورة وتحديد مناطق الإصابة..."
                ):
                    result = analyze_image(
                        image_path,
                        save_to_database=True,
                    )

                st.session_state["single_result"] = result

    with info_col:
        st.markdown(
            """
            <div class="soft-card">
                <h3 style="margin-top:0">🍓 أفضل نتيجة</h3>
                <p class="subtle">
                    استخدم صورة واضحة بإضاءة جيدة، وتجنب الصور المهزوزة
                    أو البعيدة جدًا.
                </p>
                <hr style="border:none;border-top:1px solid #f0dce4">
                <strong>النموذج</strong><br>
                YOLO11n-seg<br><br>
                <strong>حجم الإدخال</strong><br>
                640 × 640<br><br>
                <strong>حد الثقة</strong><br>
                50%
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "single_result" in st.session_state:
        st.write("")
        show_result(
            st.session_state["single_result"]
        )


elif page == "تحليل مجموعة صور":
    st.markdown(
        """
        <div class="section-title">
            <h3>تحليل مجموعة صور</h3>
            <span class="subtle">ارفع عدة صور لتحليلها دفعة واحدة</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "اختر الصور",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"تم اختيار {len(uploaded_files)} صورة.")

    if uploaded_files and st.button(
        "تحليل جميع الصور",
        use_container_width=True,
    ):
        results = []
        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            image_path = save_upload(uploaded_file)

            results.append(
                analyze_image(
                    image_path,
                    save_to_database=True,
                )
            )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        st.session_state["batch_results"] = results

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]

        table = pd.DataFrame(
            [
                {
                    "الصورة": result["image_name"],
                    "الحالة": STATUS_AR.get(result["status"]),
                    "المرض": disease_name_ar(result["disease_name"]),
                    "الثقة %": result["confidence_percentage"],
                    "المناطق": result["detected_objects"],
                }
                for result in results
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


elif page == "الإحصائيات":
    st.markdown(
        """
        <div class="section-title">
            <h3>الإحصائيات</h3>
            <span class="subtle">ملخص أداء واستخدام النظام</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    statistics = get_general_statistics()

    columns = st.columns(4)
    columns[0].metric("إجمالي الصور", statistics["total_images"])
    columns[1].metric("سليمة", statistics["healthy_images"])
    columns[2].metric("مصابة", statistics["diseased_images"])
    columns[3].metric("غير مؤكدة", statistics["uncertain_images"])

    st.write("")

    more_cols = st.columns(3)
    more_cols[0].metric(
        "متوسط الثقة",
        f"{statistics['average_confidence'] * 100:.2f}%",
    )
    more_cols[1].metric(
        "إجمالي المناطق",
        statistics["total_detected_objects"],
    )
    more_cols[2].metric(
        "متوسط زمن التحليل",
        f"{statistics['average_analysis_time'] * 1000:.0f} ms",
    )

    disease_df = get_disease_distribution()

    if not disease_df.empty:
        disease_view = disease_df.copy()
        disease_view["average_confidence"] = (
            disease_view["average_confidence"]
            .fillna(0)
            .mul(100)
            .round(2)
        )

        disease_view = disease_view.rename(
            columns={
                "disease_name": "المرض",
                "images": "عدد الصور",
                "average_confidence": "متوسط الثقة %",
            }
        )

        st.markdown("### توزيع الأمراض المكتشفة")
        st.dataframe(
            disease_view,
            use_container_width=True,
            hide_index=True,
        )


elif page == "السجل":
    st.markdown(
        """
        <div class="section-title">
            <h3>سجل التحليلات</h3>
            <span class="subtle">آخر النتائج المحفوظة في النظام</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = get_prediction_history()

    if history.empty:
        st.info("لا يوجد سجل تحليلات حتى الآن.")
    else:
        display_history = history.copy()

        display_history["confidence"] = (
            display_history["confidence"]
            .fillna(0)
            .mul(100)
            .round(2)
        )

        display_history["status"] = (
            display_history["status"]
            .map(STATUS_AR)
            .fillna(display_history["status"])
        )

        display_history["disease_name"] = (
            display_history["disease_name"]
            .fillna("-")
            .apply(disease_name_ar)
        )

        display_history = display_history.rename(
            columns={
                "id": "ID",
                "image_name": "الصورة",
                "analysis_date": "التاريخ",
                "status": "الحالة",
                "disease_name": "المرض",
                "confidence": "الثقة %",
                "detected_objects": "المناطق",
                "user_feedback": "التقييم",
            }
        )

        st.dataframe(
            display_history[
                [
                    "ID",
                    "الصورة",
                    "التاريخ",
                    "الحالة",
                    "المرض",
                    "الثقة %",
                    "المناطق",
                    "التقييم",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


elif page == "عن النظام":
    st.markdown(
        """
        <div class="soft-card">
            <h2 style="margin-top:0;color:#d92f68">عن Farmer Eye AI 🍓</h2>
            <p>
                نظام ذكي للكشف المبكر عن أمراض الفراولة باستخدام
                Instance Segmentation مع تفسير بصري للقرار عبر Grad-CAM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    cols = st.columns(4)
    cols[0].metric("النموذج", "YOLO11n-seg")
    cols[1].metric("Input Size", "640 × 640")
    cols[2].metric("Confidence", "50%")
    cols[3].metric("Classes", len(MODEL_NAMES))

    class_table = pd.DataFrame(
        [
            {
                "Class": class_name,
                "Arabic": DISEASE_AR.get(
                    class_name,
                    class_name,
                ),
            }
            for class_name in MODEL_NAMES.values()
        ]
    )

    st.markdown("### الفئات المدعومة")
    st.dataframe(
        class_table,
        use_container_width=True,
        hide_index=True,
    )


st.markdown(
    """
    <div class="footer">
        🍓 Farmer Eye AI • Strawberry Disease Detection • YOLO11n-seg
    </div>
    """,
    unsafe_allow_html=True,
)
