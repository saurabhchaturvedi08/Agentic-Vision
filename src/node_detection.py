import cv2
import numpy as np

def detect_polygons(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for filtering orange regions (adjust based on your use case)
    lower_orange = np.array([10, 100, 100])  # Lower bound of orange in HSV
    upper_orange = np.array([25, 255, 255])  # Upper bound of orange in HSV

    # Create a mask for orange regions
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate polygons from contours
    polygons = [
        cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2).tolist()
        for c in contours
    ]

    # Draw polygons on the original image (optional, for visualization)
    result_image = image.copy()
    for polygon in polygons:
        cv2.polylines(result_image, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)

    # # Save or display the resulting image (optional)
    # cv2.imwrite("polygons_detected.jpg", result_image)
    # cv2.imshow("Detected Polygons", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return polygons