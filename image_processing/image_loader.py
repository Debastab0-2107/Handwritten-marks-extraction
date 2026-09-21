import cv2
from pathlib import Path


SUPPORTED_IMAGE_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png"
}


def load_image(image_path: str) -> cv2.Mat:

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {image_path}"
        )

    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    return image