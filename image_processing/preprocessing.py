import cv2
import numpy as np

def load_and_presprocess(img: cv2.Mat) -> cv2.Mat:
    """
    Load image and return three versions:
      - img: original BGR
      - gray: grayscale
      - binary: ink=255, paper=0
    """

    if img is None:
        raise FileNotFoundError(f"Could not load the image")
    # Convert Image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    #Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=27
    )


    return  img, binary
    