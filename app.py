
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

TEMP_DIR = (
    BASE_DIR /
    "temp"
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if str(
    BASE_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            BASE_DIR
        )
    )


from modules.prediction import (
    analyze_image,
    model as prediction_model,
    MODEL_NAMES,
)

from modules.gradcam import (
    generate_gradcam,
)

from modules.analytics import (
    get_prediction_history,
    get_general_statistics,
    get_disease_distribution,
    update_prediction_feedback,
    update_prediction_notes,
)


st.set_page_config(
    page_title=
        "عين المزارع",

    layout=
        "wide",

    initial_sidebar_state=
        "expanded",
)


st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                135deg,
                #fff9fb,
                #fff2f7
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
    }

    .main-header {
        direction: rtl;
        background:
            linear-gradient(
                135deg,
                #f2447e,
                #e91e63
            );
        color: white;
        border-radius: 24px;
        padding: 28px;
        text-align: center;
        margin-bottom: 24px;
    }

    .result-card {
        direction: rtl;
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #efd3dc;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #efd3dc;
        border-radius: 16px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


STATUS_AR = {
    "Healthy":
        "سليمة",

    "Diseased":
        "مصابة",

    "Uncertain":
        "غير مؤكدة",
}


DISEASE_AR = {
    "Angular Leafspot":
        "التبقع الزاوي للأوراق",

    "Anthracnose Fruit Rot":
        "عفن الأنثراكنوز للثمار",

    "Blossom Blight":
        "لفحة الأزهار",

    "Gray Mold":
        "العفن الرمادي",

    "Healthy Strawberry":
        "فراولة سليمة",

    "Leaf Spot":
        "تبقع الأوراق",

    "Powdery Mildew Fruit":
        "البياض الدقيقي على الثمار",

    "Powdery Mildew Leaf":
        "البياض الدقيقي على الأوراق",
}


def save_upload(
    uploaded_file
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    path = (
        TEMP_DIR /
        f"{time.time_ns()}{suffix}"
    )

    path.write_bytes(
        uploaded_file.getvalue()
    )

    return path


def disease_name_ar(
    name
):

    if not name:

        return "-"

    diseases = [
        value.strip()
        for value
        in name.split(",")
        if value.strip()
    ]

    return "، ".join(
        DISEASE_AR.get(
            disease,
            disease
        )
        for disease
        in diseases
    )


def build_gradcam(
    result
):

    if (
        result[
            "status"
        ]
        ==
        "Uncertain"
    ):

        return None

    detections = (
        result[
            "detections"
        ]
    )

    if not detections:

        return None

    disease_detections = [
        detection
        for detection
        in detections
        if (
            detection[
                "class_name"
            ]
            !=
            "Healthy Strawberry"
        )
    ]

    target = max(
        (
            disease_detections
            if disease_detections
            else detections
        ),
        key=lambda item:
        item[
            "confidence"
        ]
    )

    target_class_id = int(
        target[
            "class_id"
        ]
    )

    image_path = Path(
        result[
            "original_image_path"
        ]
    )

    prediction = (
        prediction_model.predict(
            source=str(
                image_path
            ),
            conf=0.50,
            iou=0.70,
            imgsz=640,
            device=(
                0
                if torch.cuda.is_available()
                else "cpu"
            ),
            retina_masks=True,
            verbose=False
        )[0]
    )

    target_mask = None

    if (
        prediction.boxes is not None
        and prediction.masks is not None
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
            str(
                image_path
            )
        )

        height, width = (
            image.shape[:2]
        )

        indexes = np.where(
            class_ids
            ==
            target_class_id
        )[0]

        if len(
            indexes
        ):

            target_mask = np.zeros(
                (
                    height,
                    width
                ),
                dtype=np.float32
            )

            for index in indexes:

                mask = cv2.resize(
                    masks[
                        index
                    ],
                    (
                        width,
                        height
                    )
                )

                target_mask = np.maximum(
                    target_mask,
                    mask
                )

    return generate_gradcam(
        image_path=
            image_path,

        target_class_id=
            target_class_id,

        target_mask=
            target_mask,

        prediction_id=
            result[
                "prediction_id"
            ],
    )


def show_result(
    result
):

    status_ar = STATUS_AR.get(
        result[
            "status"
        ],
        result[
            "status"
        ]
    )

    disease_ar = disease_name_ar(
        result[
            "disease_name"
        ]
    )

    st.markdown(
        f"""
        <div class="result-card">
            <h2>{status_ar}</h2>
            <p>{disease_ar}</p>
            <strong>
                Confidence:
                {result['confidence_percentage']:.2f}%
            </strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    columns = st.columns(
        3
    )

    columns[0].metric(
        "درجة الثقة",
        f"{result['confidence_percentage']:.2f}%"
    )

    columns[1].metric(
        "المناطق المكتشفة",
        result[
            "detected_objects"
        ]
    )

    columns[2].metric(
        "زمن التحليل",
        f"{result['analysis_time_ms']:.1f} ms"
    )

    gradcam_result = None

    if (
        result[
            "status"
        ]
        !=
        "Uncertain"
    ):

        try:

            with st.spinner(
                "جاري إنشاء التفسير البصري..."
            ):

                gradcam_result = (
                    build_gradcam(
                        result
                    )
                )

        except Exception as error:

            st.warning(
                f"Grad-CAM unavailable: {error}"
            )

    tab1, tab2, tab3 = st.tabs(
        [
            "الصورة الأصلية",
            "Segmentation",
            "Grad-CAM",
        ]
    )

    with tab1:

        st.image(
            result[
                "original_image_path"
            ],
            use_container_width=True
        )

    with tab2:

        st.image(
            result[
                "prediction_image_path"
            ],
            use_container_width=True
        )

    with tab3:

        if gradcam_result:

            st.image(
                gradcam_result[
                    "gradcam_path"
                ],
                use_container_width=True
            )

        else:

            st.info(
                "Grad-CAM غير متاح."
            )

    prediction_id = (
        result[
            "prediction_id"
        ]
    )

    if prediction_id is not None:

        st.markdown(
            "### تقييم النتيجة"
        )

        col1, col2 = st.columns(
            2
        )

        with col1:

            if st.button(
                "النتيجة صحيحة",
                key=
                    f"correct_{prediction_id}"
            ):

                update_prediction_feedback(
                    prediction_id,
                    "Correct"
                )

                st.success(
                    "تم تسجيل التقييم."
                )

        with col2:

            if st.button(
                "النتيجة غير صحيحة",
                key=
                    f"incorrect_{prediction_id}"
            ):

                update_prediction_feedback(
                    prediction_id,
                    "Incorrect"
                )

                st.success(
                    "تم تسجيل التقييم."
                )


st.markdown(
    """
    <div class="main-header">
        <h1>عين المزارع</h1>
        <p>
            نظام ذكي للكشف المبكر عن أمراض الفراولة
            باستخدام YOLO11n-seg
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:

    page = st.radio(
        "القائمة",
        [
            "الرئيسية",
            "تحليل صورة",
            "تحليل مجموعة صور",
            "الإحصائيات",
            "السجل",
            "عن النظام",
        ]
    )


if page == "الرئيسية":

    st.markdown(
        "## نظام Farmer Eye AI"
    )

    st.write(
        """
        يقوم النظام بتحليل صور الفراولة،
        تحديد المرض ومكان الإصابة باستخدام
        Instance Segmentation،
        ثم يعرض Grad-CAM لتفسير قرار النموذج.
        """
    )


elif page == "تحليل صورة":

    uploaded_file = st.file_uploader(
        "اختر صورة فراولة",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ]
    )

    if uploaded_file:

        st.image(
            uploaded_file,
            width=450
        )

        if st.button(
            "ابدأ التحليل",
            use_container_width=True
        ):

            image_path = (
                save_upload(
                    uploaded_file
                )
            )

            with st.spinner(
                "جاري التحليل..."
            ):

                result = analyze_image(
                    image_path,
                    save_to_database=True
                )

            st.session_state[
                "single_result"
            ] = result

    if (
        "single_result"
        in st.session_state
    ):

        show_result(
            st.session_state[
                "single_result"
            ]
        )


elif page == "تحليل مجموعة صور":

    uploaded_files = st.file_uploader(
        "اختر الصور",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
        accept_multiple_files=True
    )

    if (
        uploaded_files
        and
        st.button(
            "تحليل الصور",
            use_container_width=True
        )
    ):

        results = []

        progress = st.progress(
            0
        )

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            image_path = (
                save_upload(
                    uploaded_file
                )
            )

            results.append(
                analyze_image(
                    image_path,
                    save_to_database=True
                )
            )

            progress.progress(
                (
                    index + 1
                ) /
                len(
                    uploaded_files
                )
            )

        st.session_state[
            "batch_results"
        ] = results

    if (
        "batch_results"
        in st.session_state
    ):

        results = (
            st.session_state[
                "batch_results"
            ]
        )

        table = pd.DataFrame(
            [
                {
                    "الصورة":
                        result[
                            "image_name"
                        ],

                    "الحالة":
                        STATUS_AR.get(
                            result[
                                "status"
                            ]
                        ),

                    "المرض":
                        disease_name_ar(
                            result[
                                "disease_name"
                            ]
                        ),

                    "الثقة %":
                        result[
                            "confidence_percentage"
                        ],
                }
                for result
                in results
            ]
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True
        )


elif page == "الإحصائيات":

    statistics = (
        get_general_statistics()
    )

    columns = st.columns(
        4
    )

    columns[0].metric(
        "إجمالي الصور",
        statistics[
            "total_images"
        ]
    )

    columns[1].metric(
        "سليمة",
        statistics[
            "healthy_images"
        ]
    )

    columns[2].metric(
        "مصابة",
        statistics[
            "diseased_images"
        ]
    )

    columns[3].metric(
        "غير مؤكدة",
        statistics[
            "uncertain_images"
        ]
    )

    disease_df = (
        get_disease_distribution()
    )

    if not disease_df.empty:

        st.markdown(
            "### الأمراض المكتشفة"
        )

        st.dataframe(
            disease_df,
            use_container_width=True,
            hide_index=True
        )


elif page == "السجل":

    history = (
        get_prediction_history()
    )

    if history.empty:

        st.info(
            "لا يوجد سجل تحليلات."
        )

    else:

        display_history = (
            history.copy()
        )

        display_history[
            "confidence"
        ] = (
            display_history[
                "confidence"
            ]
            .fillna(0)
            * 100
        ).round(
            2
        )

        st.dataframe(
            display_history[
                [
                    "id",
                    "image_name",
                    "analysis_date",
                    "status",
                    "disease_name",
                    "confidence",
                    "detected_objects",
                    "user_feedback",
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


elif page == "عن النظام":

    st.markdown(
        "## Farmer Eye AI"
    )

    st.write(
        """
        Model: YOLO11n-seg

        Input Size: 640 x 640

        Confidence Threshold: 50%

        Explainability: Multi-Layer Grad-CAM

        Database: SQLite
        """
    )

    class_table = pd.DataFrame(
        [
            {
                "Class":
                    class_name,

                "Arabic":
                    DISEASE_AR.get(
                        class_name,
                        class_name
                    ),
            }
            for class_name
            in MODEL_NAMES.values()
        ]
    )

    st.dataframe(
        class_table,
        use_container_width=True,
        hide_index=True
    )
