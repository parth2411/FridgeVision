import torch
import csv
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def crop_object(image, object_coords):
    img_width, img_height = image.size
    x_center, y_center, width, height = object_coords
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    left = x_center - (width / 2)
    top = y_center - (height / 2)
    right = x_center + (width / 2)
    bottom = y_center + (height / 2)
    return image.crop((left, top, right, bottom))

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def read_objects_info(csv_file_path):
    objects_info = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            objects_info.append({
                'coords': (float(row['X']), float(row['Y']), float(row['Width']), float(row['Height'])),
                'name': row['Item Name']
            })
    return objects_info

def calculate_metrics(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)
    ssim_value = ssim(image1, image2, win_size=3, multichannel=True)
    psnr_value = psnr(image1, image2)
    return ssim_value, psnr_value

def display_images_with_highlights(image_path1, image_path2, changed_areas):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    axs[0].imshow(image1)
    axs[1].imshow(image2)
    for area in changed_areas:
        img_width, img_height = image1.size
        x_center, y_center, width, height = area
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        left = x_center - (width / 2)
        top = y_center - (height / 2)
        rect1 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect1)
        axs[1].add_patch(rect2)
    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

def compare_objects(image_path1, image_path2, csv_file_path):
    objects_info = read_objects_info(csv_file_path)
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    model = models.resnet18(pretrained=True)
    model.eval()
    changed_areas = []

    for object_info in objects_info:
        object_coords, object_name = object_info['coords'], object_info['name']
        cropped_image1 = crop_object(image1, object_coords)
        cropped_image2 = crop_object(image2, object_coords)
        image_tensor1 = preprocess_image(cropped_image1)
        image_tensor2 = preprocess_image(cropped_image2)
        feature1 = model(image_tensor1)
        feature2 = model(image_tensor2)
        similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
        print(f"{object_name} Cosine similarity: {similarity.item()}")

        if similarity.item() < 0.9:
            print(f"Significant change detected in {object_name}.")
            changed_areas.append(object_coords)
        else:
            print(f"No significant change detected in {object_name}.")

        ssim_value, psnr_value = calculate_metrics(cropped_image1, cropped_image2)
        print(f"{object_name} SSIM: {ssim_value}, PSNR: {psnr_value}")

    if changed_areas:
        display_images_with_highlights(image_path1, image_path2, changed_areas)
    else:
        print("No significant changes detected between the images.")

# Example usage
image_path1 = 'out.png'
image_path2 = 'out (2).png'
csv_file_path = 'result.csv'
compare_objects(image_path1, image_path2, csv_file_path)
