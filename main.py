import cv2

from image_processing.preprocessing import (
    load_and_preprocess
)

from image_processing.detection import (
    find_ink_components,
    filter_components,
    draw_components,
    group_into_lines,
    get_line_boxes
)


# ==========================================
# STEP 1: Load image
# ==========================================

image_path = "input/image3.jpeg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load: {image_path}"
    )

image_width = image.shape[1]
image_height = image.shape[0]

img_width = 900
img_height = int(image_height * (img_width / image_width))

img = cv2.resize(
    image,
    (img_width, img_height)
)

# ==========================================
# STEP 2: Preprocess
# ==========================================

img, gray, binary = load_and_preprocess(img)


# ==========================================
# STEP 3: Find connected components
# ==========================================

components = find_ink_components(binary)

print(
    "Components detected:",
    len(components)
)


# ==========================================
# STEP 4: Remove small noise
# ==========================================

components = filter_components(
    components,
    min_area=20
)

print(
    "Components after filtering:",
    len(components)
)


# ==========================================
# STEP 5: Draw components
# ==========================================

component_image = draw_components(
    img,
    components
)


# ==========================================
# STEP 6: Group into lines
# ==========================================

lines = group_into_lines(
    components,
    y_tolerance=20
)

print(
    "Lines detected:",
    len(lines)
)


# ==========================================
# STEP 7: Get line bounding boxes
# ==========================================

line_boxes = get_line_boxes(lines)


# ==========================================
# STEP 8: Print positions
# ==========================================

print("\nDetected text regions:")

for index, box in enumerate(line_boxes):

    x, y, w, h = box

    print(
        f"Line {index + 1}: "
        f"x={x}, "
        f"y={y}, "
        f"width={w}, "
        f"height={h}"
    )


# ==========================================
# STEP 9: Draw line boxes
# ==========================================

line_image = img.copy()

for index, (x, y, w, h) in enumerate(
    line_boxes
):

    cv2.rectangle(
        line_image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        line_image,
        f"Line {index + 1}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )


# ==========================================
# STEP 10: Save results
# ==========================================

cv2.imwrite(
    "binary.jpg",
    binary
)

cv2.imwrite(
    "detected_components.jpg",
    component_image
)

cv2.imwrite(
    "detected_text_lines.jpg",
    line_image
)


# ==========================================
# STEP 11: Display
# ==========================================

cv2.imshow(
    "Original",
    img
)

cv2.imshow(
    "Binary",
    binary
)

cv2.imshow(
    "Components",
    component_image
)

cv2.imshow(
    "Detected Text Lines",
    line_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()


