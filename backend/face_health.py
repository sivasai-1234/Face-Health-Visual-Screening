from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXTS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MODEL_PATH = Path(__file__).with_name("face_health_model.joblib")

DISCLAIMER = (
    "This is a preliminary visual screening aid, NOT a medical diagnosis. "
    "Only a qualified doctor and appropriate laboratory tests can diagnose "
    "a medical condition."
)


INDICATORS = {
    "scleral_yellowing": {
        "label": "Yellowish eye whites",
        "meaning": (
            "Yellowish eye whites can have several causes. Lighting may "
            "also create a false appearance."
        ),
        "advice": (
            "If the yellow colour persists, arrange a medical consultation. "
            "A doctor may recommend bilirubin and liver-function tests."
        ),
    },
    "eye_redness": {
        "label": "Red or irritated eye whites",
        "meaning": (
            "This may occur because of eye strain, dryness, allergies, "
            "irritation, or conjunctivitis."
        ),
        "advice": (
            "Avoid rubbing your eyes. Seek medical care if you have pain, "
            "discharge, or changes in vision."
        ),
    },
    "pallor": {
        "label": "Pale lips or low colour contrast",
        "meaning": (
            "Pale lips may sometimes occur with low haemoglobin or iron "
            "deficiency. Lighting and skin tone can affect this result."
        ),
        "advice": (
            "A doctor may recommend a CBC or haemoglobin blood test "
            "if anaemia is suspected."
        ),
    },
    "dark_circles": {
        "label": "Dark under-eye area",
        "meaning": (
            "Dark areas under the eyes may be related to tiredness, "
            "sleep loss, allergies, dehydration, or genetics."
        ),
        "advice": (
            "Try to maintain regular sleep, drink enough water, and "
            "reduce late-night screen time."
        ),
    },
    "lip_dryness": {
        "label": "Dry or rough lip texture",
        "meaning": (
            "Dry lips may occur because of low fluid intake, dry weather, "
            "or frequent lip-licking."
        ),
        "advice": (
            "Drink water regularly and use lip balm. Consult a doctor "
            "if cracking or sores continue."
        ),
    },
}


FEATURE_KEYS = [
    "cheek_L",
    "cheek_a",
    "cheek_b",
    "lip_L",
    "lip_a",
    "lip_b",
    "lip_texture",
    "sclera_L",
    "sclera_a",
    "sclera_b",
    "under_L",
    "under_a",
    "under_b",
    "red_ratio",
    "yellow_ratio",
    "lip_contrast",
    "under_ratio",
]


L_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]

R_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

LIPS_OUT = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

LIPS_IN = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]

IRIS_A = [468, 469, 470, 471, 472]
IRIS_B = [473, 474, 475, 476, 477]


def load_image(
    path: str | Path,
    max_side: int = 1600,
) -> np.ndarray:
    """Load an image safely, including Unicode paths on Windows."""

    image_path = Path(path)

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unsupported, empty, or corrupted image.")

    height, width = image.shape[:2]

    if max(height, width) > max_side:
        scale = max_side / max(height, width)

        image = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    return image


def save_image(
    path: str | Path,
    image: np.ndarray,
) -> bool:
    """Save an image safely, including Unicode paths on Windows."""

    output_path = Path(path)
    extension = output_path.suffix.lower() or ".jpg"

    if extension == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        extension = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    success, encoded = cv2.imencode(extension, image, params)

    if success:
        encoded.tofile(str(output_path))

    return bool(success)


def hull_mask(
    shape: tuple[int, int, int],
    points: np.ndarray,
) -> np.ndarray:
    """Create a convex-hull mask from facial landmark points."""

    mask = np.zeros(shape[:2], dtype=np.uint8)
    points = np.asarray(points, dtype=np.int32)

    if len(points) >= 3:
        cv2.fillConvexPoly(
            mask,
            cv2.convexHull(points),
            255,
        )

    return mask


def circle_mask(
    shape: tuple[int, int, int],
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Create a circular region mask."""

    mask = np.zeros(shape[:2], dtype=np.uint8)

    cv2.circle(
        mask,
        (
            int(center[0]),
            int(center[1]),
        ),
        max(1, int(radius)),
        255,
        -1,
    )

    return mask


def box_mask(
    shape: tuple[int, int, int],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    """Create a safe, clipped rectangular mask."""

    height, width = shape[:2]

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    x0 = int(max(0, x0))
    x1 = int(min(width, x1))
    y0 = int(max(0, y0))
    y1 = int(min(height, y1))

    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255

    return mask


class FaceAnalyzer:
    """
    Detect facial regions using MediaPipe FaceMesh.

    OpenCV Haar detection is used as a limited fallback.
    """

    def __init__(self) -> None:
        self.mesh = None
        self.method = "haar"

        try:
            import mediapipe as mp

            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

            self.method = "mediapipe"

        except Exception as error:
            print(
                f"[info] MediaPipe unavailable: {error}. "
                "Using limited Haar fallback."
            )

        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        if self.haar.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar face detector."
            )

    def release(self) -> None:
        """Release MediaPipe resources safely."""

        if self.mesh is not None:
            self.mesh.close()

    def regions(
        self,
        image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Find the face and facial screening regions."""

        if self.mesh is not None:
            mesh_result = self._mesh_regions(image)

            if mesh_result is not None:
                return mesh_result

        return self._haar_regions(image)

    def _mesh_regions(
        self,
        image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Extract facial regions using MediaPipe landmarks."""

        height, width = image.shape[:2]

        result = self.mesh.process(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )
        )

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark

        points = np.array(
            [
                [
                    point.x * width,
                    point.y * height,
                ]
                for point in landmarks
            ],
            dtype=np.float32,
        )

        x0 = points[:, 0].min()
        y0 = points[:, 1].min()
        x1 = points[:, 0].max()
        y1 = points[:, 1].max()

        face_width = float(
            np.linalg.norm(
                points[234] - points[454]
            )
        )

        lips_outer = hull_mask(
            image.shape,
            points[LIPS_OUT],
        )

        lips_inner = hull_mask(
            image.shape,
            points[LIPS_IN],
        )

        lips = cv2.subtract(
            lips_outer,
            cv2.dilate(
                lips_inner,
                np.ones(
                    (3, 3),
                    dtype=np.uint8,
                ),
            ),
        )

        iris_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        for iris_indices in (IRIS_A, IRIS_B):
            iris_center = points[iris_indices[0]]

            iris_radius = max(
                np.linalg.norm(
                    points[index] - iris_center
                )
                for index in iris_indices[1:]
            ) * 1.2

            iris_mask = cv2.bitwise_or(
                iris_mask,
                circle_mask(
                    image.shape,
                    iris_center,
                    iris_radius,
                ),
            )

        kernel_size = max(
            3,
            int(face_width * 0.012),
        ) | 1

        eye_kernel = np.ones(
            (
                kernel_size,
                kernel_size,
            ),
            dtype=np.uint8,
        )

        sclera = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        under_eye = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        for eye_indices in (L_EYE, R_EYE):
            eye = hull_mask(
                image.shape,
                points[eye_indices],
            )

            eye = cv2.erode(
                eye,
                eye_kernel,
            )

            sclera = cv2.bitwise_or(
                sclera,
                cv2.subtract(
                    eye,
                    cv2.dilate(
                        iris_mask,
                        np.ones(
                            (3, 3),
                            dtype=np.uint8,
                        ),
                    ),
                ),
            )

            eye_x0, eye_y0 = points[eye_indices].min(
                axis=0
            )

            eye_x1, eye_y1 = points[eye_indices].max(
                axis=0
            )

            eye_width = eye_x1 - eye_x0

            under_eye = cv2.bitwise_or(
                under_eye,
                box_mask(
                    image.shape,
                    eye_x0 + 0.15 * eye_width,
                    eye_y1 + 0.12 * eye_width,
                    eye_x1 - 0.15 * eye_width,
                    eye_y1 + 0.40 * eye_width,
                ),
            )

        cheeks = cv2.bitwise_or(
            circle_mask(
                image.shape,
                points[205],
                face_width * 0.07,
            ),
            circle_mask(
                image.shape,
                points[425],
                face_width * 0.07,
            ),
        )

        return {
            "method": "mediapipe",
            "bbox": (
                x0,
                y0,
                x1,
                y1,
            ),
            "face_w": face_width,
            "lips": lips,
            "sclera": sclera,
            "under": under_eye,
            "cheek": cheeks,
        }

    def _haar_regions(
        self,
        image: np.ndarray,
    ) -> dict[str, Any] | None:
        """Use approximate facial regions with Haar detection."""

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) == 0:
            return None

        x, y, face_width, face_height = max(
            faces,
            key=lambda face: face[2] * face[3],
        )

        def relative_box(
            left: float,
            top: float,
            right: float,
            bottom: float,
        ) -> np.ndarray:
            return box_mask(
                image.shape,
                x + left * face_width,
                y + top * face_height,
                x + right * face_width,
                y + bottom * face_height,
            )

        cheek_left = relative_box(
            0.20,
            0.50,
            0.38,
            0.62,
        )

        cheek_right = relative_box(
            0.62,
            0.50,
            0.80,
            0.62,
        )

        cheeks = cv2.bitwise_or(
            cheek_left,
            cheek_right,
        )

        lips = relative_box(
            0.35,
            0.72,
            0.65,
            0.84,
        )

        empty_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        return {
            "method": "haar",
            "bbox": (
                float(x),
                float(y),
                float(x + face_width),
                float(y + face_height),
            ),
            "face_w": float(face_width),
            "lips": lips,
            "sclera": empty_mask,
            "under": empty_mask,
            "cheek": cheeks,
        }


def check_quality(
    image: np.ndarray,
    regions: dict[str, Any],
) -> dict[str, Any]:
    """Check whether the image is suitable for screening."""

    x0, y0, x1, y1 = [
        int(value)
        for value in regions["bbox"]
    ]

    face_crop = image[
        max(0, y0):min(image.shape[0], y1),
        max(0, x0):min(image.shape[1], x1),
    ]

    if face_crop.size == 0:
        return {
            "status": "reject",
            "messages": [
                "Could not extract a valid face region. "
                "Please retake the photo."
            ],
        }

    if regions["face_w"] < 110:
        return {
            "status": "reject",
            "messages": [
                "Face is too small. Move closer to the camera "
                "or use a clearer image."
            ],
        }

    resized_height = max(
        10,
        int(
            256
            * face_crop.shape[0]
            / max(face_crop.shape[1], 1)
        ),
    )

    gray = cv2.cvtColor(
        cv2.resize(
            face_crop,
            (256, resized_height),
        ),
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(gray.mean())
    clipped_ratio = float(
        (gray > 250).mean()
    )

    messages: list[str] = []
    status = "ok"

    if sharpness < 6:
        return {
            "status": "reject",
            "blur": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "messages": [
                "The image is too blurry. Hold still, improve "
                "lighting, and wait for camera focus."
            ],
        }

    if sharpness < 25:
        status = "warn"
        messages.append(
            "The image is slightly blurry, so results may "
            "be less reliable."
        )

    if brightness < 60:
        status = "warn"
        messages.append(
            "The face is under-exposed. Retake the photo "
            "in brighter light."
        )

    if brightness > 210 or clipped_ratio > 0.15:
        status = "warn"
        messages.append(
            "The face is over-exposed or has glare. "
            "Avoid direct flash."
        )

    if regions["method"] == "haar":
        status = "warn"
        messages.append(
            "Limited mode is active. Eye-based screening "
            "requires MediaPipe."
        )

    return {
        "status": status,
        "blur": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "messages": messages,
    }


def median_lab(
    lab_image: np.ndarray,
    mask: np.ndarray,
    keep_bright: bool = False,
) -> np.ndarray | None:
    """Calculate median LAB values inside a masked region."""

    pixels = lab_image[mask > 0]

    if len(pixels) < 20:
        return None

    if keep_bright:
        cutoff = np.percentile(
            pixels[:, 0],
            50,
        )

        pixels = pixels[
            pixels[:, 0] >= cutoff
        ]

    if len(pixels) == 0:
        return None

    return np.median(
        pixels,
        axis=0,
    )


def extract_features(
    image: np.ndarray,
    regions: dict[str, Any],
) -> dict[str, float | None]:
    """Extract colour and texture features."""

    lab = cv2.cvtColor(
        image.astype(np.float32) / 255.0,
        cv2.COLOR_BGR2LAB,
    )

    features: dict[str, float | None] = {
        key: None
        for key in FEATURE_KEYS
    }

    cheeks = median_lab(
        lab,
        regions["cheek"],
    )

    lips = median_lab(
        lab,
        regions["lips"],
    )

    sclera = median_lab(
        lab,
        regions["sclera"],
        keep_bright=True,
    )

    under_eye = median_lab(
        lab,
        regions["under"],
    )

    if cheeks is not None:
        features["cheek_L"] = round(
            float(cheeks[0]),
            3,
        )

        features["cheek_a"] = round(
            float(cheeks[1]),
            3,
        )

        features["cheek_b"] = round(
            float(cheeks[2]),
            3,
        )

    if lips is not None:
        features["lip_L"] = round(
            float(lips[0]),
            3,
        )

        features["lip_a"] = round(
            float(lips[1]),
            3,
        )

        features["lip_b"] = round(
            float(lips[2]),
            3,
        )

        sigma = max(
            1.0,
            regions["face_w"] * 0.006,
        )

        high_pass = lab[..., 0] - cv2.GaussianBlur(
            lab[..., 0],
            (0, 0),
            sigma,
        )

        lip_pixels = high_pass[
            regions["lips"] > 0
        ]

        if len(lip_pixels) > 0:
            features["lip_texture"] = round(
                float(lip_pixels.std()),
                3,
            )

    if sclera is not None:
        features["sclera_L"] = round(
            float(sclera[0]),
            3,
        )

        features["sclera_a"] = round(
            float(sclera[1]),
            3,
        )

        features["sclera_b"] = round(
            float(sclera[2]),
            3,
        )

    if under_eye is not None:
        features["under_L"] = round(
            float(under_eye[0]),
            3,
        )

        features["under_a"] = round(
            float(under_eye[1]),
            3,
        )

        features["under_b"] = round(
            float(under_eye[2]),
            3,
        )

    if cheeks is not None:
        if sclera is not None:
            features["red_ratio"] = round(
                float(
                    sclera[1]
                    / max(cheeks[1], 4.0)
                ),
                3,
            )

            features["yellow_ratio"] = round(
                float(
                    sclera[2]
                    / max(cheeks[2], 5.0)
                ),
                3,
            )

        if lips is not None:
            features["lip_contrast"] = round(
                float(
                    lips[1] - cheeks[1]
                ),
                3,
            )

        if under_eye is not None:
            features["under_ratio"] = round(
                float(
                    under_eye[0]
                    / max(cheeks[0], 1.0)
                ),
                3,
            )

    return features


def ramp(
    value: float | None,
    low: float,
    high: float,
) -> float | None:
    """Convert a feature value into a score between 0 and 1."""

    if value is None:
        return None

    denominator = high - low

    if denominator == 0:
        return None

    score = (value - low) / denominator

    return float(
        np.clip(score, 0.0, 1.0)
    )


def score_indicators(
    features: dict[str, float | None],
) -> list[dict[str, Any]]:
    """
    Generate educational visual screening scores.

    These thresholds are not clinically validated and do not
    diagnose any medical condition.
    """

    raw_scores = {
        "scleral_yellowing": ramp(
            features["yellow_ratio"],
            0.55,
            1.00,
        ),
        "eye_redness": ramp(
            features["red_ratio"],
            0.55,
            1.05,
        ),
        "pallor": ramp(
            features["lip_contrast"],
            6.0,
            1.0,
        ),
        "dark_circles": ramp(
            features["under_ratio"],
            0.92,
            0.78,
        ),
        "lip_dryness": ramp(
            features["lip_texture"],
            1.2,
            3.0,
        ),
    }

    scored_indicators: list[dict[str, Any]] = []

    for indicator_id, score in raw_scores.items():
        if score is None:
            continue

        if score >= 0.65:
            level = "high"
        elif score >= 0.35:
            level = "moderate"
        else:
            level = "low"

        scored_indicators.append(
            {
                "id": indicator_id,
                "score": round(score, 2),
                "level": level,
                **INDICATORS[indicator_id],
            }
        )

    return sorted(
        scored_indicators,
        key=lambda item: item["score"],
        reverse=True,
    )


def create_health_alert(
    indicators: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Create a clear alert for the frontend.

    This is not a medical diagnosis. It only responds to
    visual screening indicators.
    """

    urgent_signs = {
        "scleral_yellowing",
    }

    urgent_indicators = [
        item
        for item in indicators
        if item["id"] in urgent_signs
        and item["level"] == "high"
    ]

    if urgent_indicators:
        return {
            "level": "urgent_review",
            "title": "Medical attention recommended",
            "message": (
                "A potentially concerning visual sign was noticed. "
                "Please arrange a medical consultation promptly."
            ),
            "action": (
                "A qualified doctor should examine you and decide "
                "whether medical tests are necessary. If you have "
                "severe symptoms or feel seriously unwell, seek "
                "emergency medical care immediately."
            ),
        }

    concerning_indicators = [
        item
        for item in indicators
        if item["level"] in {
            "high",
            "moderate",
        }
    ]

    if concerning_indicators:
        return {
            "level": "attention",
            "title": "Some signs need attention",
            "message": (
                "Some visible changes were noticed. These signs can "
                "have different causes and do not confirm a disease."
            ),
            "action": (
                "If the changes continue or concern you, consult "
                "a qualified healthcare professional."
            ),
        }

    return {
        "level": "normal_review",
        "title": "Screening completed",
        "message": (
            "No major visible concerns were detected in this image."
        ),
        "action": (
            "This result does not rule out medical conditions."
        ),
    }


def model_predict(
    features: dict[str, float | None],
    model_path: str | Path = MODEL_PATH,
) -> list[dict[str, Any]] | None:
    """Use the optional trained Joblib model."""

    path = Path(model_path)

    if not path.exists():
        return None

    try:
        import joblib

        bundle = joblib.load(path)

        values = [
            np.nan
            if features.get(key) is None
            else features[key]
            for key in bundle["keys"]
        ]

        probabilities = bundle["model"].predict_proba(
            np.array(
                [values],
                dtype=float,
            )
        )[0]

        predictions = [
            {
                "class": str(label),
                "probability": round(
                    float(probability),
                    3,
                ),
            }
            for label, probability in zip(
                bundle["model"].classes_,
                probabilities,
            )
        ]

        return sorted(
            predictions,
            key=lambda item: item["probability"],
            reverse=True,
        )

    except Exception as error:
        return [
            {
                "error": (
                    "The trained model could not be used: "
                    f"{str(error)}"
                )
            }
        ]


def draw_overlay(
    image: np.ndarray,
    regions: dict[str, Any],
    indicators: list[dict[str, Any]],
) -> np.ndarray:
    """Create an annotated image for optional backend use."""

    output = image.copy()

    region_colours = {
        "lips": (0, 0, 255),
        "sclera": (255, 200, 0),
        "under": (0, 200, 255),
        "cheek": (0, 255, 0),
    }

    for region_name, colour in region_colours.items():
        mask = regions[region_name]

        if mask is not None and mask.any():
            layer = output.copy()
            layer[mask > 0] = colour

            output = cv2.addWeighted(
                layer,
                0.40,
                output,
                0.60,
                0,
            )

    x0, y0, x1, y1 = [
        int(value)
        for value in regions["bbox"]
    ]

    cv2.rectangle(
        output,
        (x0, y0),
        (x1, y1),
        (0, 185, 135),
        2,
    )

    flagged = [
        item
        for item in indicators
        if item["level"] != "low"
    ][:3]

    lines = []

    if flagged:
        for item in flagged:
            lines.append(
                f"{item['label']}: "
                f"{item['level'].upper()}"
            )
    else:
        lines.append(
            "No notable visual signs detected"
        )

    y_position = 28

    for line in lines:
        cv2.putText(
            output,
            line,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            line,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y_position += 25

    return output


def analyze_image(
    path: str | Path,
    analyzer: FaceAnalyzer,
    out_dir: str | Path,
    model_path: str | Path = MODEL_PATH,
    show: bool = False,
) -> dict[str, Any]:
    """Run the complete face-screening pipeline."""

    image_path = Path(path)
    output_dir = Path(out_dir)

    report: dict[str, Any] = {
        "file": str(image_path),
        "filename": image_path.name,
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "disclaimer": DISCLAIMER,
    }

    try:
        image = load_image(image_path)

    except Exception as error:
        report.update(
            {
                "status": "error",
                "error": (
                    f"Could not read image: {error}"
                ),
                "error_type": "IMAGE_READ_ERROR",
            }
        )

        return report

    regions = analyzer.regions(image)

    if regions is None:
        report.update(
            {
                "status": "rejected",
                "error": (
                    "No face detected. Use a clear, "
                    "front-facing image with good lighting."
                ),
                "error_type": "NO_FACE_DETECTED",
            }
        )

        return report

    quality = check_quality(
        image,
        regions,
    )

    report["quality"] = quality

    if quality["status"] == "reject":
        report.update(
            {
                "status": "rejected",
                "error": " ".join(
                    quality["messages"]
                ),
                "error_type": "IMAGE_QUALITY_REJECTED",
            }
        )

        return report

    try:
        features = extract_features(
            image,
            regions,
        )

        indicators = score_indicators(
            features
        )

        prediction = model_predict(
            features,
            model_path,
        )

        health_alert = create_health_alert(
            indicators
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        annotated_image = draw_overlay(
            image,
            regions,
            indicators,
        )

        annotated_path = (
            output_dir
            / f"{image_path.stem}_annotated.jpg"
        )

        save_image(
            annotated_path,
            annotated_image,
        )

        report.update(
            {
                "status": "ok",
                "method": regions["method"],
                "features": features,
                "indicators": indicators,
                "model_prediction": prediction,
                "health_alert": health_alert,
                "annotated_image": str(
                    annotated_path
                ),
            }
        )

        report_path = (
            output_dir
            / f"{image_path.stem}_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if show and has_display():
            cv2.imshow(
                "Face Health Screening",
                annotated_image,
            )

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return report

    except Exception as error:
        report.update(
            {
                "status": "error",
                "error": (
                    f"Analysis failed: {error}"
                ),
                "error_type": "ANALYSIS_ERROR",
            }
        )

        return report


def has_display() -> bool:
    """Check whether OpenCV GUI windows are available."""

    return (
        os.name == "nt"
        or sys.platform == "darwin"
        or bool(
            os.environ.get("DISPLAY")
        )
    )


if __name__ == "__main__":
    print(
        "Face Health Screening core module "
        "loaded successfully."
    )

    print(
        "Use fastapi_server.py to start the web API."
    )