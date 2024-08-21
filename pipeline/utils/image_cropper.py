import os
import cv2
import numpy as np

def crop_and_center_image(image_path, target_size=(256, 256), lesion_proportion=0.8):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for detecting dark lesions (I haven't found the best ranges still)
    lower = np.array([0, 20, 20])
    upper = np.array([179, 255, 255])

    # Create a mask to detect the lesion based on color
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, skip this image
    if not contours:
        print(f"No contours found in image: {image_path}")
        return None

    # Find the largest contour which is assumed to be the lesion
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the required size for the lesion to occupy 80% of the target size
    lesion_target_size = int(target_size[0] * lesion_proportion)

    # Calculate the scaling factor to resize the lesion to the target lesion size
    scale_factor = lesion_target_size / max(w, h)

    # Resize the original image to ensure the lesion fits within the target lesion size
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Get the new dimensions after resizing
    new_h, new_w = resized_image.shape[:2]

    # Calculate the center of the lesion in the resized image
    center_x, center_y = int((x + w / 2) * scale_factor), int((y + h / 2) * scale_factor)

    # Calculate the cropping coordinates to center the lesion
    crop_x1 = max(0, center_x - target_size[0] // 2)
    crop_y1 = max(0, center_y - target_size[1] // 2)
    crop_x2 = min(new_w, center_x + target_size[0] // 2)
    crop_y2 = min(new_h, center_y + target_size[1] // 2)

    # Adjust the crop coordinates if necessary to maintain the target size
    if crop_x2 - crop_x1 < target_size[0]:
        crop_x1 = max(0, crop_x2 - target_size[0])
    if crop_y2 - crop_y1 < target_size[1]:
        crop_y1 = max(0, crop_y2 - target_size[1])

    # Ensure the cropped area is exactly the target size
    cropped_image = resized_image[crop_y1:crop_y2, crop_x1:crop_x2]

    # If the crop size is not exactly the target size, resize it to fit
    if cropped_image.shape[0] != target_size[0] or cropped_image.shape[1] != target_size[1]:
        cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)

    return cropped_image