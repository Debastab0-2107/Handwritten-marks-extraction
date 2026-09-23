import cv2
import numpy as np


import cv2
import numpy as np


def detect_text_lines_projection(
    binary: cv2.Mat,
    min_ink_ratio: float = 0.03,
    min_line_height: int = 8,
    max_gap: int = 3
):
    """
    Detect individual handwritten text lines
    using a horizontal projection profile.
    """

    height, width = binary.shape

    # ------------------------------------------------
    # 1. Calculate ink ratio for every Y row
    # ------------------------------------------------

    ink_pixels = np.sum(
        binary > 0,
        axis=1
    )

    ink_ratio = ink_pixels / width

    # ------------------------------------------------
    # 2. Smooth the projection
    # ------------------------------------------------

    # Convert to float32 because OpenCV filtering
    # expects a suitable numeric type.
    projection = ink_ratio.astype(
        np.float32
    )

    projection = cv2.GaussianBlur(
        projection.reshape(-1, 1),
        (1, 7),
        0
    ).flatten()

    # ------------------------------------------------
    # 3. Threshold the smoothed projection
    # ------------------------------------------------

    active = projection >= min_ink_ratio

    # ------------------------------------------------
    # 4. Find active vertical regions
    # ------------------------------------------------

    ranges = []

    start = None
    gap = 0

    for y in range(height):

        if active[y]:

            if start is None:
                start = y

            gap = 0

        else:

            if start is not None:

                gap += 1

                if gap > max_gap:

                    end = y - gap

                    if (
                        end - start + 1
                        >= min_line_height
                    ):
                        ranges.append(
                            (start, end)
                        )

                    start = None
                    gap = 0

    # ------------------------------------------------
    # 5. Handle region at bottom
    # ------------------------------------------------

    if start is not None:

        end = height - 1

        if (
            end - start + 1
            >= min_line_height
        ):
            ranges.append(
                (start, end)
            )

    # ------------------------------------------------
    # 6. Create bounding boxes
    # ------------------------------------------------

    line_boxes = []

    for y1, y2 in ranges:

        region = binary[
            y1:y2 + 1,
            :
        ]

        ys, xs = np.where(
            region > 0
        )

        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        x2 = int(xs.max())

        line_boxes.append(
            (
                x1,
                y1,
                x2 - x1 + 1,
                y2 - y1 + 1
            )
        )

    return line_boxes


def calculate_horizontal_projection(binary):

    height, width = binary.shape

    ink_pixels = np.sum(
        binary > 0,
        axis=1
    )

    ink_ratio = ink_pixels / width

    return ink_ratio

# done on 23rd september 2026
import cv2
import numpy as np


def detect_table_lines(binary: cv2.Mat):
    """
    Detect horizontal and vertical lines
    present in tables.
    """

    # ----------------------------------
    # Horizontal lines
    # ----------------------------------

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40, 1)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    # ----------------------------------
    # Vertical lines
    # ----------------------------------

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 40)
    )

    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    return horizontal_lines, vertical_lines

def build_table_structure(horizontal_lines, vertical_lines):
    """
    Combine horizontal and vertical table lines
    to create a complete table structure.
    """

    table_structure = cv2.add(horizontal_lines, vertical_lines)

    return table_structure

def detect_table_regions(table_structure):
    """
    Detect large rectangular table regions
    from the combined table structure.

    Returns:
        regions: list of (x, y, width, height)
    """

    # Make the detected lines slightly thicker.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    connected = cv2.dilate(
        table_structure,
        kernel,
        iterations=1
    )

    # Find connected regions
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        # Ignore very small regions
        if w < 150 or h < 50:
            continue

        regions.append(
            (x, y, w, h)
        )

    return regions