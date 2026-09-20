import cv2
from image_processing.preprocessing import load_and_presprocess

image_path = "input/image1.jpeg"

image = cv2.imread(image_path)

if image is None:
    print("Could not load the image.")
    exit()

print("Image loaded successfully!")


print("Original size:")
print("Width:", image.shape[1])
print("Height:", image.shape[0])

# Resize only for display
display_width = 900

scale = display_width / image.shape[1]
display_height = int(image.shape[0] * scale)

display_image = cv2.resize(
    image,
    (display_width, display_height)
)

# cv2.imshow("Original Image", display_image)

binary, binary2 = load_and_presprocess(display_image)

cv2.imshow("IMAGE 1 ", binary)
cv2.imshow("IMAGE 2", binary2)
cv2.waitKey(0)
cv2.destroyAllWindows()