import cv2
import numpy as np


def rotate_image(
    image: cv2.Mat,
    angle: float
) -> cv2.Mat:
    """
    Rotate image around its center.
    """

    height, width = image.shape[:2]

    center = (
        width / 2,
        height / 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def calculate_projection_score(
    binary: cv2.Mat
) -> float:
    """
    Calculate how strongly the ink is
    concentrated into horizontal rows.

    Higher score means stronger horizontal
    text-line structure.
    """

    row_ink = np.sum(
        binary > 0,
        axis=1
    )

    score = np.var(row_ink)

    return float(score)


def find_best_skew_angle(
    binary: cv2.Mat,
    min_angle: float = -5.0,
    max_angle: float = 5.0,
    step: float = 0.25
) -> float:
    """
    Test several rotation angles and find
    the angle that gives the strongest
    horizontal text structure.
    """

    best_angle = 0.0
    best_score = -1.0

    angles = np.arange(
        min_angle,
        max_angle + step,
        step
    )

    for angle in angles:

        rotated = rotate_image(
            binary,
            angle
        )

        score = calculate_projection_score(
            rotated
        )

        if score > best_score:

            best_score = score
            best_angle = angle

    return float(best_angle)


def deskew_image(
    img: cv2.Mat
) -> cv2.Mat:
    """
    Automatically correct small rotation/skew
    in the handwritten page.
    """

    # -----------------------------------------
    # 1. Convert to grayscale
    # -----------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------
    # 2. Create temporary binary image
    # -----------------------------------------

    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # -----------------------------------------
    # 3. Remove small noise
    # -----------------------------------------

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # -----------------------------------------
    # 4. Find best angle
    # -----------------------------------------

    angle = find_best_skew_angle(
        binary,
        min_angle=-5,
        max_angle=5,
        step=0.25
    )

    print(
        f"Detected skew angle: {angle:.2f} degrees"
    )

    # -----------------------------------------
    # 5. Rotate original color image
    # -----------------------------------------

    deskewed = rotate_image(
        img,
        angle
    )

    return deskewed