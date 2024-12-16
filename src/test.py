import cv2
import numpy as np
from src.node_detection import detect_polygons
from src.polygon_correlation import correlate_polygons
from src.generate_response import generate_response

def correlate_images(image1_path, image2_path):
    """
    Correlate two images using ORB (Oriented FAST and Rotated BRIEF) feature detection and matching.
    
    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        dict: A dictionary with the similarity score, number of matches, and the matched image.
    """
    # Load the images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return {"error": "One or both image paths are invalid."}

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher object with Hamming distance and cross-checking enabled
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance indicates better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Compute a similarity score (normalized by the number of keypoints in the smaller image)
    similarity_score = len(matches) / min(len(kp1), len(kp2))

    # Draw the top matches
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return {
        "similarity_score": similarity_score,
        "num_matches": len(matches),
        "matched_image": matched_image
    }

# Example usage
if __name__ == "__main__":
    # Paths to your images
    image1_path = "images\image_1.jpg"
    image2_path = "images\image_2.jpg"
    
    polygons_one = detect_polygons(image1_path)
    print("nodes of polygon one: \n",polygons_one)
    polygons_two = detect_polygons(image2_path)
    print("nodes of polygon two: \n", polygons_two)

    similarity = correlate_polygons(image1_path, image2_path)
    print("similarity: \n", similarity)
    
    question = "What are the nodes of the orange polygon in both images? "
    generate_response(question, image1_path, image2_path, similarity)

    # # Correlate images
    # result = correlate_images(image1_path, image2_path)

    # if "error" in result:
    #     print(result["error"])
    # else:
    #     print(f"Similarity Score: {result['similarity_score']:.2f}")
    #     print(f"Number of Matches: {result['num_matches']}")

    #     # Display the matched image
    #     cv2.imshow("Matched Image", result["matched_image"])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
