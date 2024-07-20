import torch
import csv
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.spatial.distance import euclidean

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

def encode_image(model, image_tensor):
    with torch.no_grad():
        return model(image_tensor)

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

def display_images_with_highlights(image_path1, image_path2, changed_areas):
    # Load the images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Display the images
    axs[0].imshow(image1)
    axs[1].imshow(image2)

    # Highlight changed areas
    for area in changed_areas:
        # Calculate rectangle parameters
        img_width, img_height = image1.size
        x_center, y_center, width, height = area
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        left = x_center - (width / 2)
        top = y_center - (height / 2)

        # Create a rectangle patch for the first image
        rect1 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Create a rectangle patch for the second image
        rect2 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the plots
        axs[0].add_patch(rect1)
        axs[1].add_patch(rect2)

    axs[0].axis('off')
    axs[1].axis('off')
    plt.show()

def plot_euclidean_distances(distances):
    objects = [item[0] for item in distances]
    values = [item[1] for item in distances]

    plt.figure(figsize=(12, 6))
    plt.barh(objects, values, color='skyblue')
    plt.xlabel('Euclidean Distance')
    plt.title('Euclidean Distances of Objects')
    plt.gca().invert_yaxis()  # Highest distance at the top
    plt.show()

def compare_objects(image_path1, image_path2, csv_file_path):
    objects_info = read_objects_info(csv_file_path)
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    model = models.resnet18(pretrained=True)
    model.eval()
    # Remove the final layer to get the latent features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    changed_areas = []
    euclidean_distances = []

    for object_info in objects_info:
        object_coords, object_name = object_info['coords'], object_info['name']
        cropped_image1 = crop_object(image1, object_coords)
        cropped_image2 = crop_object(image2, object_coords)
        image_tensor1 = preprocess_image(cropped_image1)
        image_tensor2 = preprocess_image(cropped_image2)
        feature1 = encode_image(model, image_tensor1).squeeze().numpy()
        feature2 = encode_image(model, image_tensor2).squeeze().numpy()
        distance = euclidean(feature1, feature2)
        euclidean_distances.append((object_name, distance))
        print(f"{object_name} Euclidean distance: {distance}")

        if distance > 1.0:  # Assuming a distance threshold
            print(f"Significant change detected in {object_name}.")
            changed_areas.append(object_coords)
        else:
            print(f"No significant change detected in {object_name}.")

    # Sort by distance
    euclidean_distances.sort(key=lambda x: x[1], reverse=True)

    # Print sorted distances
    for name, distance in euclidean_distances:
        print(f"{name}: {distance}")

    # Display images with highlighted changes if any
    if changed_areas:
        display_images_with_highlights(image_path1, image_path2, changed_areas)
    else:
        print("No significant changes detected between the images.")

    # Plot the Euclidean distances
    plot_euclidean_distances(euclidean_distances)

# Example usage
image_path1 = 'out.png'
image_path2 = 'out (2).png'
csv_file_path = 'result.csv'
compare_objects(image_path1, image_path2, csv_file_path)
