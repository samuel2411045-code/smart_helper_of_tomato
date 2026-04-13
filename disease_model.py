from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CLASSES = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]


@dataclass(frozen=True)
class DiseasePrediction:
    label: str
    confidence: float
    remedy: str
    model_type: str = "CNN"


REMEDIES: dict[str, str] = {
    "Early_blight": "Apply fungicides like Mancozeb; ensure proper crop rotation and remove infected debris.",
    "Healthy": "The plant is healthy! Continue regular monitoring and balanced organic fertilization.",
    "Late_blight": "Use fungicides (e.g., Chlorothalonil); improve air circulation and avoid overhead watering.",
    "Leaf Miner": "Use Neem oil or specialized insecticides; introduce beneficial insects like parasitic wasps.",
    "Magnesium Deficiency": "Apply Epsom salt (Magnesium Sulfate) to the soil or as a foliar spray.",
    "Nitrogen Deficiency": "Apply Nitrogen-rich fertilizers like Urea, Compost, or Blood Meal.",
    "Pottassium Deficiency": "Add Potassium-rich fertilizers like Muriate of Potash (MOP) or wood ash.",
    "Spotted Wilt Virus": "Control thrips (the vector) using insecticides; remove and destroy infected plants immediately.",
}


def _load_keras_model(model_path: Path):
    # Lazy import so the app can run without TensorFlow installed (or during light dev).
    from tensorflow.keras.models import load_model  # type: ignore

    return load_model(str(model_path))



def predict_leaf_disease(
    *,
    image_rgb: np.ndarray,
    model_path: str | Path = "models/disease_model.h5",
    class_names: list[str] | None = None,
    input_size: tuple[int, int] = (224, 224),
    model_type: str = "CNN",
) -> DiseasePrediction:
    """
    Predict tomato leaf disease using a Keras `.h5` model (CNN or ViT) if present.
    If the model file is missing, returns a safe placeholder prediction.
    """

    class_names = class_names or DEFAULT_CLASSES
    model_path = Path(model_path)

    if not model_path.exists():
        return DiseasePrediction(
            label="Model not found",
            confidence=0.0,
            remedy=f"Add your trained {model_type} model at `{model_path.as_posix()}` to enable predictions.",
            model_type=model_type,
        )

    # Basic preprocessing: resize + scale to [0,1]
    import cv2

    img = cv2.resize(image_rgb, input_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    x = np.expand_dims(img, axis=0)

    try:
        model = _load_keras_model(model_path)
        probs = model.predict(x, verbose=0)[0]
    except Exception as e:
        return DiseasePrediction(
            label="Error Loading Model",
            confidence=0.0,
            remedy=f"Error loading {model_type} model: {e}",
            model_type=model_type,
        )

    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    conf = float(probs[idx]) if hasattr(probs, "__len__") else float(probs)
    remedy = REMEDIES.get(label, "Consult local agriculture office for diagnosis and treatment guidance.")

    return DiseasePrediction(label=label, confidence=conf, remedy=remedy, model_type=model_type)

