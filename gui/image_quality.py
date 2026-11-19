from dataclasses import dataclass

import numpy as np
from PIL import Image

MIN_SHORT_SIDE = 256
MIN_PIXELS = 100_000

MIN_BRIGHTNESS = 50.0

CLIPPED_LEVEL = 250
MAX_CLIPPED_FRACTION = 0.5

MIN_SHARPNESS = 50.0

MAX_ANALYSIS_SIDE = 512


@dataclass(frozen=True)
class QualityIssue:
    code: str
    title: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    width: int
    height: int
    brightness: float
    clipped_fraction: float
    sharpness: float | None
    issues: tuple[QualityIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _flatten(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _mean_luminance(gray: Image.Image) -> float:
    return float(np.asarray(gray, dtype=np.float32).mean())


def _clipped_fraction(gray: Image.Image) -> float:
    data = np.asarray(gray, dtype=np.float32)
    return float(np.mean(data >= CLIPPED_LEVEL))


def _sharpness(gray: Image.Image) -> float:
    sample = gray.copy()
    sample.thumbnail((MAX_ANALYSIS_SIDE, MAX_ANALYSIS_SIDE), Image.LANCZOS)
    data = np.asarray(sample, dtype=np.float32)
    if data.shape[0] < 3 or data.shape[1] < 3:
        return 0.0
    center = data[1:-1, 1:-1]
    laplacian = (
        4 * center
        - data[:-2, 1:-1]
        - data[2:, 1:-1]
        - data[1:-1, :-2]
        - data[1:-1, 2:]
    )
    return float(laplacian.var())


def assess_image(image: Image.Image) -> QualityReport:
    rgb = _flatten(image)
    width, height = rgb.size
    gray = rgb.convert("L")
    brightness = _mean_luminance(gray)
    clipped = _clipped_fraction(gray)

    issues: list[QualityIssue] = []

    if min(width, height) < MIN_SHORT_SIDE or width * height < MIN_PIXELS:
        issues.append(
            QualityIssue(
                code="low_resolution",
                title="Low resolution",
                message=(
                    f"This photo is {width}×{height} px. Use a photo at least "
                    f"{MIN_SHORT_SIDE} px on the shorter side."
                ),
            )
        )

    too_dark = brightness < MIN_BRIGHTNESS
    too_bright = clipped > MAX_CLIPPED_FRACTION

    if too_dark:
        issues.append(
            QualityIssue(
                code="too_dark",
                title="Too dark",
                message=(
                    "The photo is underexposed. Move to better light or use "
                    "the flash, then take another photo."
                ),
            )
        )
    elif too_bright:
        issues.append(
            QualityIssue(
                code="too_bright",
                title="Too bright",
                message=(
                    "The photo is overexposed. Avoid direct flash or harsh "
                    "sunlight, then take another photo."
                ),
            )
        )

    sharpness = None
    if not too_dark and not too_bright:
        sharpness = _sharpness(gray)
        if sharpness < MIN_SHARPNESS:
            issues.append(
                QualityIssue(
                    code="blurry",
                    title="Blurry photo",
                    message=(
                        "The photo looks out of focus. Hold the camera steady "
                        "and let it focus on the leaf."
                    ),
                )
            )

    return QualityReport(
        width=width,
        height=height,
        brightness=brightness,
        clipped_fraction=clipped,
        sharpness=sharpness,
        issues=tuple(issues),
    )
