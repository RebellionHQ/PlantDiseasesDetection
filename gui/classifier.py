import io
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
WEIGHTS_PATH = MODEL_DIR / "weights" / "model.pth"

_api = None


def _get_api():
    global _api
    if _api is None:
        if str(MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL_DIR))
        from model_api import plant_api

        _api = plant_api
    return _api


def load_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def to_jpeg(image: Image.Image, max_side: int, quality: int = 82) -> bytes:
    preview = image.copy()
    preview.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def split_label(label: str) -> tuple[str, str]:
    if "___" in label:
        plant, disease = label.split("___", 1)
    else:
        plant, disease = "Unknown", label
    return _pretty(plant), _pretty(disease)


def _pretty(text: str) -> str:
    text = " ".join(text.replace("_", " ").split())
    if not text:
        return text
    return text[0].upper() + text[1:]


def is_healthy(disease: str) -> bool:
    return disease.strip().lower() == "healthy"


def confidence_color(value: float) -> str:
    if value >= 75:
        return "#2F7D4F"
    if value >= 45:
        return "#C77700"
    return "#B23A2E"


def predict(image: Image.Image) -> dict:
    raw = _get_api()(image, weights=str(WEIGHTS_PATH))
    plant, disease = split_label(raw["class"])

    top = []
    for item in raw["top_results"]:
        item_plant, item_disease = split_label(item["class"])
        top.append(
            {
                "plant": item_plant,
                "disease": item_disease,
                "confidence": float(item["conf"]),
            }
        )

    return {
        "plant": plant,
        "disease": disease,
        "confidence": float(raw["confidence"]),
        "top": top,
    }
