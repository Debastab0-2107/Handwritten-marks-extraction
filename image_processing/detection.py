import cv2

def find_ink_components(binary_image: cv2.Mat) -> list:
    """Find connected white-pixel regions.
        Returns a list of : 
            (x, y, width, height, area)
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8,
        )
    )

    components = []
    #Label 0 is the background, so we start from 1
    for i in range(1, num_labels):
        x      = stats[i, cv2.CC_STAT_LEFT]
        y      = stats[i, cv2.CC_STAT_TOP]
        width  = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area   = stats[i, cv2.CC_STAT_AREA]
        components.append((x, y, width, height, area))

    return components

def filter_components(components: list, min_area: int = 155) -> list:
    """Filter out components that are too small."""
    filtered = []

    for component in components:
        x, y, width, height, area = component
        if area >= min_area:
            filtered.append(component)

    return filtered

def draw_components(image: cv2.Mat, components: list) -> cv2.Mat:
    """Draw bounding boxes around components on the image."""
    output =image.copy()
    for component in components:
        x, y, width, height, area = component
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return output    

def group_into_lines(
    components,
    y_tolerance=20
):
    """
    Group components having similar vertical positions.
    """

    components = sorted(
        components,
        key=lambda item: item[1]
    )

    lines = []

    for component in components:

        x, y, w, h, area = component

        center_y = y + h / 2

        placed = False

        for line in lines:

            line_center_y = sum(
                item[1] + item[3] / 2
                for item in line
            ) / len(line)

            if abs(
                center_y - line_center_y
            ) <= y_tolerance:

                line.append(component)
                placed = True
                break

        if not placed:

            lines.append(
                [component]
            )

    # Left -> right
    for line in lines:

        line.sort(
            key=lambda item: item[0]
        )

    return lines

def get_line_boxes(lines):

    line_boxes = []

    for line in lines:

        if not line:
            continue

        x1 = min(
            item[0]
            for item in line
        )

        y1 = min(
            item[1]
            for item in line
        )

        x2 = max(
            item[0] + item[2]
            for item in line
        )

        y2 = max(
            item[1] + item[3]
            for item in line
        )

        line_boxes.append(
            (
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )
        )

    return line_boxes

