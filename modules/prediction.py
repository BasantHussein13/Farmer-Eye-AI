
import shutil
import sqlite3
import time

from datetime import datetime
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "best.pt"

DATABASE_PATH = (
    BASE_DIR /
    "database" /
    "farmer_eye.db"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.70
IMAGE_SIZE = 640

HEALTHY_CLASS_NAME = (
    "Healthy Strawberry"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


DEVICE = (
    0
    if torch.cuda.is_available()
    else "cpu"
)

model = YOLO(
    str(MODEL_PATH)
)

MODEL_NAMES = model.names

if isinstance(
    MODEL_NAMES,
    list
):

    MODEL_NAMES = {
        i: str(name)
        for i, name
        in enumerate(
            MODEL_NAMES
        )
    }

else:

    MODEL_NAMES = {
        int(i): str(name)
        for i, name
        in MODEL_NAMES.items()
    }


HEALTHY_CLASS_IDS = {
    class_id
    for class_id, class_name
    in MODEL_NAMES.items()
    if (
        class_name.strip().lower()
        ==
        HEALTHY_CLASS_NAME.lower()
    )
}


def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def save_prediction(
    image_name,
    status,
    disease_name,
    confidence,
    detected_objects,
    analysis_time,
    segmentation_image,
    original_image
):

    connection = (
        get_connection()
    )

    cursor = (
        connection.cursor()
    )

    cursor.execute(
        """
        INSERT INTO prediction_history
        (
            image_name,
            analysis_date,
            status,
            disease_name,
            confidence,
            detected_objects,
            analysis_time,
            segmentation_image,
            gradcam_image,
            original_image,
            user_feedback,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_name,
            datetime.now().isoformat(
                timespec="seconds"
            ),
            status,
            disease_name,
            float(confidence),
            int(detected_objects),
            float(analysis_time),
            str(segmentation_image),
            None,
            str(original_image),
            None,
            None,
        )
    )

    prediction_id = (
        cursor.lastrowid
    )

    connection.commit()
    connection.close()

    return prediction_id


def validate_image(
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            image_path
        )

    if (
        image_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):

        raise ValueError(
            "Unsupported image format."
        )

    return image_path


def parse_prediction(
    result
):

    records = []
    healthy = []
    diseases = []

    if (
        result.boxes is None
        or len(
            result.boxes
        ) == 0
    ):

        return {
            "status": "Uncertain",
            "status_ar": "غير مؤكدة",
            "disease_name": None,
            "confidence": 0.0,
            "detected_objects": 0,
            "records": [],
        }

    class_ids = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    for index, (
        class_id,
        confidence
    ) in enumerate(
        zip(
            class_ids,
            confidences
        )
    ):

        confidence = float(
            confidence
        )

        if (
            confidence
            <
            CONFIDENCE_THRESHOLD
        ):
            continue

        class_name = (
            MODEL_NAMES[
                int(
                    class_id
                )
            ]
        )

        record = {
            "instance_number":
                index + 1,

            "class_id":
                int(
                    class_id
                ),

            "class_name":
                class_name,

            "confidence":
                confidence,

            "confidence_percentage":
                round(
                    confidence * 100,
                    2
                ),
        }

        records.append(
            record
        )

        if (
            int(
                class_id
            )
            in HEALTHY_CLASS_IDS
        ):

            healthy.append(
                record
            )

        else:

            diseases.append(
                record
            )

    if diseases:

        strongest = max(
            diseases,
            key=lambda item:
            item[
                "confidence"
            ]
        )

        disease_names = sorted({
            item[
                "class_name"
            ]
            for item
            in diseases
        })

        return {
            "status":
                "Diseased",

            "status_ar":
                "مصابة",

            "disease_name":
                ", ".join(
                    disease_names
                ),

            "confidence":
                strongest[
                    "confidence"
                ],

            "detected_objects":
                len(
                    diseases
                ),

            "records":
                records,
        }

    if healthy:

        strongest = max(
            healthy,
            key=lambda item:
            item[
                "confidence"
            ]
        )

        return {
            "status":
                "Healthy",

            "status_ar":
                "سليمة",

            "disease_name":
                None,

            "confidence":
                strongest[
                    "confidence"
                ],

            "detected_objects":
                len(
                    healthy
                ),

            "records":
                records,
        }

    return {
        "status":
            "Uncertain",

        "status_ar":
            "غير مؤكدة",

        "disease_name":
            None,

        "confidence":
            0.0,

        "detected_objects":
            0,

        "records":
            [],
    }


def analyze_image(
    image_path,
    save_to_database=True
):

    image_path = (
        validate_image(
            image_path
        )
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    stem = (
        f"{image_path.stem}_"
        f"{timestamp}"
    )

    original_path = (
        OUTPUT_DIR /
        (
            f"{stem}_original"
            f"{image_path.suffix.lower()}"
        )
    )

    prediction_path = (
        OUTPUT_DIR /
        f"{stem}_prediction.jpg"
    )

    start_time = (
        time.perf_counter()
    )

    result = model.predict(
        source=str(
            image_path
        ),
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        retina_masks=True,
        verbose=False
    )[0]

    analysis_time = (
        time.perf_counter()
        -
        start_time
    )

    parsed = (
        parse_prediction(
            result
        )
    )

    shutil.copy2(
        image_path,
        original_path
    )

    annotated = (
        result.plot(
            boxes=True,
            labels=True,
            conf=True,
            masks=True
        )
    )

    cv2.imwrite(
        str(
            prediction_path
        ),
        annotated
    )

    prediction_id = None

    if save_to_database:

        prediction_id = (
            save_prediction(
                image_name=
                    image_path.name,

                status=
                    parsed[
                        "status"
                    ],

                disease_name=
                    parsed[
                        "disease_name"
                    ],

                confidence=
                    parsed[
                        "confidence"
                    ],

                detected_objects=
                    parsed[
                        "detected_objects"
                    ],

                analysis_time=
                    analysis_time,

                segmentation_image=
                    prediction_path,

                original_image=
                    original_path,
            )
        )

    return {
        "prediction_id":
            prediction_id,

        "image_name":
            image_path.name,

        "status":
            parsed[
                "status"
            ],

        "status_ar":
            parsed[
                "status_ar"
            ],

        "disease_name":
            parsed[
                "disease_name"
            ],

        "confidence":
            float(
                parsed[
                    "confidence"
                ]
            ),

        "confidence_percentage":
            round(
                float(
                    parsed[
                        "confidence"
                    ]
                ) * 100,
                2
            ),

        "detected_objects":
            int(
                parsed[
                    "detected_objects"
                ]
            ),

        "analysis_time":
            float(
                analysis_time
            ),

        "analysis_time_ms":
            round(
                analysis_time *
                1000,
                2
            ),

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "original_image_path":
            str(
                original_path
            ),

        "prediction_image_path":
            str(
                prediction_path
            ),

        "detections":
            parsed[
                "records"
            ],
    }
