import cv2
import matplotlib.pyplot as plt
from image_processing.image_loader import load_image
from image_processing.preprocessing import (
    load_and_preprocess
)

from image_processing.deskew import (
    deskew_image
)

from image_processing.detection import (
    calculate_horizontal_projection,
    detect_text_lines_projection
)


# ==========================================
# STEP 1: Load image
# ==========================================

image_path = "input/image4.png"

image = load_image(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {image_path}"
    )


# ==========================================
# STEP 2: Resize image
# ==========================================

image_width = image.shape[1]
image_height = image.shape[0]

img_width = 900

img_height = int(
    image_height *
    (img_width / image_width)
)

img = cv2.resize(
    image,
    (img_width, img_height)
)


# ==========================================
# STEP 3: Deskew
# ==========================================

print("\nStarting deskew...")

deskewed = deskew_image(
    img
)

print("Deskew completed.")


# ==========================================
# STEP 4: Preprocess
# ==========================================

img, gray, binary = load_and_preprocess(
    deskewed
)


# ==========================================
# STEP 5: Detect text lines
# ==========================================

line_boxes = detect_text_lines_projection(
    binary,
    min_ink_ratio=0.03,
    min_line_height=8,
    max_gap=3
)


# ==========================================
# STEP 6: Print detected lines
# ==========================================

print(
    "\nText lines detected:",
    len(line_boxes)
)

print("\nDetected regions:")

for i, (x, y, w, h) in enumerate(
    line_boxes
):

    print(
        f"Line {i + 1}: "
        f"x={x}, "
        f"y={y}, "
        f"width={w}, "
        f"height={h}"
    )


# ==========================================
# STEP 7: Draw boxes
# ==========================================

output = deskewed.copy()

for i, (x, y, w, h) in enumerate(
    line_boxes
):

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        f"Line {i + 1}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


# ==========================================
# STEP 8: Save results
# ==========================================

cv2.imwrite(
    "deskewed.jpg",
    deskewed
)

cv2.imwrite(
    "binary.jpg",
    binary
)

cv2.imwrite(
    "detected_text_lines.jpg",
    output
)


# ==========================================
# STEP 9: Display images
# ==========================================

cv2.imshow(
    "Original",
    img
)

cv2.imshow(
    "Deskewed",
    deskewed
)

cv2.imshow(
    "Binary",
    binary
)

cv2.imshow(
    "Detected Text Lines",
    output
)


# ==========================================
# STEP 10: Horizontal projection
# ==========================================

projection = calculate_horizontal_projection(
    binary
)


# ==========================================
# STEP 11: Plot projection
# ==========================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    projection,
    range(len(projection))
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Ink Ratio"
)

plt.ylabel(
    "Y Position"
)

plt.title(
    "Horizontal Projection Profile"
)

plt.show()


# ==========================================
# STEP 12: Wait
# ==========================================

cv2.waitKey(0)

cv2.destroyAllWindows()