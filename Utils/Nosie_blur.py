import os
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np

# Define the directories for the images

image_dir = '/Users/parthbhalodiya/Desktop/aug'

# Define the augmenters
blur_augmenter = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 1.5))  # Blur up to 1.5 pixels
])

noise_augmenter = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(0, 0.0094 * 255))  # Noise up to 0.94% of pixel values
])


# Function to create a mosaic image
def create_mosaic(images):
    h, w, _ = images[0].shape
    mosaic = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    mosaic[0:h, 0:w, :] = images[0]
    mosaic[0:h, w:2 * w, :] = images[1]
    mosaic[h:2 * h, 0:w, :] = images[2]
    mosaic[h:2 * h, w:2 * w, :] = images[3]
    return mosaic


# Function to augment and save images
def augment_save(image_path, augmenter, augmentation_type, outputs_per_example):
    image = Image.open(image_path)
    image_np = np.array(image)

    for i in range(outputs_per_example):
        # Apply augmentation
        image_aug = augmenter(image=image_np)
        image_aug_pil = Image.fromarray(image_aug)

        # Construct new filename for the augmented image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        new_image_path = os.path.join(image_dir, f"{base_name}_{augmentation_type}_{i}.jpg")

        # Save augmented image
        image_aug_pil.save(new_image_path)


# Get all image paths
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(".jpg")]

# Process each file in the directory
for image_path in image_paths:
    # Apply and save blur augmentations
    augment_save(image_path, blur_augmenter, "blur", 3)
    # Apply and save noise augmentations
    augment_save(image_path, noise_augmenter, "noise", 3)

# Apply and save mosaic augmentations
# Create mosaics from groups of 4 images
for i in range(0, len(image_paths), 4):
    if i + 3 < len(image_paths):
        images = [np.array(Image.open(image_paths[j])) for j in range(i, i + 4)]
        mosaic = create_mosaic(images)
        mosaic_pil = Image.fromarray(mosaic)

        # Construct new filename for the mosaic image
        base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
        new_image_path = os.path.join(image_dir, f"{base_name}_mosaic.jpg")

        # Save mosaic image
        mosaic_pil.save(new_image_path)

print("Augmentation complete.")
