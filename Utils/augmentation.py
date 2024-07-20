import cv2
import os
import numpy as np


def horizontal_flip(image):
    return cv2.flip(image, 1)


def scale_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)


def motion_blur(image, kernel_size=15):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)


def color_manipulation(image, brightness=0, contrast=1.0, saturation=1.0):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)

    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 2] = hsv[:, :, 2] * contrast + brightness

    hsv = np.clip(hsv, 0, 255)
    return cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)


def add_fog(image, fog_intensity=0.2):
    fog = np.ones_like(image) * 255
    fog = fog.astype(np.uint8)
    return cv2.addWeighted(image, 1 - fog_intensity, fog, fog_intensity, 0)


# Set the path to your image folder
image_folder_path = '/Users/parthbhalodiya/Desktop/1'

# Set the path to the output folder
output_folder_path = '/Users/parthbhalodiya/Desktop/1'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through all images in the folder
for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)

        # Read the image
        original_image = cv2.imread(image_path)

        # Apply transformations
        flipped_image = horizontal_flip(original_image)
        #scaled_image = scale_image(original_image, scale_factor=0.8)
        blurred_image = motion_blur(original_image)
        manipulated_image = color_manipulation(original_image, brightness=20, contrast=1.2, saturation=1.5)
        foggy_image = add_fog(original_image, fog_intensity=0.3)

        # Save the transformed images to the output folder
        cv2.imwrite(os.path.join(output_folder_path, 'flipped_' + filename), flipped_image)
        #cv2.imwrite(os.path.join(output_folder_path, 'scaled_' + filename), scaled_image)
        cv2.imwrite(os.path.join(output_folder_path, 'blurred_' + filename), blurred_image)
        cv2.imwrite(os.path.join(output_folder_path, 'manipulated_' + filename), manipulated_image)
        cv2.imwrite(os.path.join(output_folder_path, 'foggy_' + filename), foggy_image)

print("Image processing complete.")
