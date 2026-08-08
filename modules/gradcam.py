
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR /
    "model" /
    "best.pt"
)

DATABASE_PATH = (
    BASE_DIR /
    "database" /
    "farmer_eye.db"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs" /
    "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMAGE_SIZE = 640

TARGET_LAYER_INDEXES = [
    16,
    19,
    22
]

CAM_THRESHOLD = 0.30

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


yolo_model = YOLO(
    str(
        MODEL_PATH
    )
)

core_model = (
    yolo_model.model
    .to(
        DEVICE
    )
    .eval()
)

for parameter in (
    core_model.parameters()
):

    parameter.requires_grad_(
        True
    )


MODEL_NAMES = (
    yolo_model.names
)

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

NUMBER_OF_CLASSES = len(
    MODEL_NAMES
)


target_layers = [
    (
        f"layer_{index}",
        core_model.model[
            index
        ]
    )
    for index
    in TARGET_LAYER_INDEXES
    if index < len(
        core_model.model
    )
]


def collect_tensors(
    value
):

    if torch.is_tensor(
        value
    ):
        return [
            value
        ]

    tensors = []

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        for item in value:

            tensors.extend(
                collect_tensors(
                    item
                )
            )

    elif isinstance(
        value,
        dict
    ):

        for item in (
            value.values()
        ):

            tensors.extend(
                collect_tensors(
                    item
                )
            )

    return tensors


def normalize_cam(
    cam
):

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam = np.nan_to_num(
        cam
    )

    cam -= (
        cam.min()
    )

    maximum = float(
        cam.max()
    )

    if maximum <= 1e-12:

        return np.zeros_like(
            cam
        )

    return np.clip(
        cam / maximum,
        0,
        1
    )


def prepare_image(
    image_bgr
):

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = (
        image_rgb.shape[:2]
    )

    scale = min(
        IMAGE_SIZE / width,
        IMAGE_SIZE / height
    )

    new_width = int(
        round(
            width * scale
        )
    )

    new_height = int(
        round(
            height * scale
        )
    )

    resized = cv2.resize(
        image_rgb,
        (
            new_width,
            new_height
        )
    )

    canvas = np.full(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ),
        114,
        dtype=np.uint8
    )

    left = (
        IMAGE_SIZE -
        new_width
    ) // 2

    top = (
        IMAGE_SIZE -
        new_height
    ) // 2

    canvas[
        top:
        top + new_height,

        left:
        left + new_width
    ] = resized

    return (
        canvas,
        left,
        top,
        new_width,
        new_height
    )


def update_database(
    prediction_id,
    path
):

    if prediction_id is None:
        return

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.execute(
        """
        UPDATE prediction_history
        SET gradcam_image = ?
        WHERE id = ?
        """,
        (
            str(
                path
            ),
            int(
                prediction_id
            )
        )
    )

    connection.commit()
    connection.close()


def generate_gradcam(
    image_path,
    target_class_id,
    target_mask=None,
    prediction_id=None
):

    image_path = Path(
        image_path
    )

    image_bgr = cv2.imread(
        str(
            image_path
        )
    )

    if image_bgr is None:

        raise ValueError(
            "Cannot read image."
        )

    original_height, original_width = (
        image_bgr.shape[:2]
    )

    (
        prepared,
        left,
        top,
        resized_width,
        resized_height
    ) = prepare_image(
        image_bgr
    )

    input_tensor = (
        torch.from_numpy(
            prepared.copy()
        )
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
        .float()
        / 255.0
    )

    input_tensor = (
        input_tensor
        .to(
            DEVICE
        )
        .detach()
        .clone()
    )

    input_tensor.requires_grad_(
        True
    )

    features = {}
    hooks = []

    def build_hook(
        name
    ):

        def hook(
            module,
            inputs,
            output
        ):

            tensors = [
                tensor
                for tensor
                in collect_tensors(
                    output
                )
                if (
                    tensor.ndim == 4
                    and tensor.requires_grad
                )
            ]

            if not tensors:
                return

            activation = max(
                tensors,
                key=lambda tensor:
                tensor.numel()
            )

            features[
                name
            ] = {
                "activation":
                    activation
            }

            activation.register_hook(
                lambda gradient:
                features[
                    name
                ].update({
                    "gradient":
                        gradient
                })
            )

        return hook

    for name, layer in (
        target_layers
    ):

        hooks.append(
            layer.register_forward_hook(
                build_hook(
                    name
                )
            )
        )

    try:

        core_model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():

            raw_output = (
                core_model(
                    input_tensor
                )
            )

            predictions = [
                tensor
                for tensor
                in collect_tensors(
                    raw_output
                )
                if (
                    tensor.ndim == 3
                    and tensor.requires_grad
                )
            ]

            prediction_tensor = max(
                predictions,
                key=lambda tensor:
                tensor.numel()
            )

            if (
                prediction_tensor.shape[1]
                >
                prediction_tensor.shape[2]
            ):

                prediction_tensor = (
                    prediction_tensor.transpose(
                        1,
                        2
                    )
                )

            class_scores = (
                prediction_tensor[
                    :,
                    4:
                    4 + NUMBER_OF_CLASSES,
                    :
                ]
            )

            scores = (
                class_scores[
                    :,
                    int(
                        target_class_id
                    ),
                    :
                ]
                .reshape(-1)
            )

            top_k = min(
                20,
                int(
                    scores.numel()
                )
            )

            target_score = (
                torch.topk(
                    scores,
                    k=top_k
                )
                .values
                .mean()
            )

            target_score.backward()

        cams = []

        for name, _ in (
            target_layers
        ):

            data = (
                features.get(
                    name
                )
            )

            if not data:
                continue

            activation = (
                data.get(
                    "activation"
                )
            )

            gradient = (
                data.get(
                    "gradient"
                )
            )

            if (
                activation is None
                or gradient is None
            ):
                continue

            activation = (
                activation[0]
            )

            gradient = (
                gradient[0]
            )

            weights = (
                gradient.mean(
                    dim=(
                        1,
                        2
                    ),
                    keepdim=True
                )
            )

            cam = torch.relu(
                (
                    weights *
                    activation
                ).sum(
                    dim=0
                )
            )

            cam = (
                cam
                .detach()
                .cpu()
                .numpy()
            )

            cam = normalize_cam(
                cam
            )

            cam = cv2.resize(
                cam,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

            cam = cam[
                top:
                top + resized_height,

                left:
                left + resized_width
            ]

            cam = cv2.resize(
                cam,
                (
                    original_width,
                    original_height
                )
            )

            cams.append(
                normalize_cam(
                    cam
                )
            )

        if not cams:

            raise RuntimeError(
                "Grad-CAM failed."
            )

        gradcam_map = normalize_cam(
            np.mean(
                np.stack(
                    cams
                ),
                axis=0
            )
        )

        gradcam_map = np.where(
            gradcam_map
            >=
            CAM_THRESHOLD,
            gradcam_map,
            0
        )

        gradcam_map = normalize_cam(
            gradcam_map
        )

        if target_mask is not None:

            target_mask = cv2.resize(
                np.asarray(
                    target_mask,
                    dtype=np.float32
                ),
                (
                    original_width,
                    original_height
                )
            )

            target_mask = normalize_cam(
                target_mask
            )

            gradcam_map = normalize_cam(
                gradcam_map *
                (
                    0.10 +
                    0.90 *
                    target_mask
                )
            )

        heatmap = cv2.applyColorMap(
            np.uint8(
                gradcam_map *
                255
            ),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(
            image_bgr,
            0.70,
            heatmap,
            0.30,
            0
        )

        output_path = (
            OUTPUT_DIR /
            (
                f"{image_path.stem}_"
                f"{prediction_id or 'temp'}_"
                f"gradcam.jpg"
            )
        )

        cv2.imwrite(
            str(
                output_path
            ),
            overlay
        )

        update_database(
            prediction_id,
            output_path
        )

        return {
            "gradcam_path":
                str(
                    output_path
                ),

            "target_class_id":
                int(
                    target_class_id
                ),

            "target_class_name":
                MODEL_NAMES[
                    int(
                        target_class_id
                    )
                ],

            "target_layers":
                TARGET_LAYER_INDEXES,

            "target_score":
                float(
                    target_score
                    .detach()
                    .cpu()
                ),
        }

    finally:

        for hook in hooks:

            hook.remove()

        core_model.zero_grad(
            set_to_none=True
        )
