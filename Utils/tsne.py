import torch
import csv
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import numpy as np


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


def visualize_with_pca_tsne(features1, features2, labels, changed_labels):
    # Combine features and apply PCA
    features = torch.cat((features1, features2), dim=0).numpy()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    # Determine the number of samples
    num_samples = features.shape[0]

    # Set perplexity to be less than the number of samples
    perplexity = min(10, num_samples - 1)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    tsne_result = tsne.fit_transform(features)

    # Generate distinct colors for each object name
    unique_labels = list(set(labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    # Plot PCA results
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(labels):
        idx = unique_labels.index(label)
        color = colors(idx)
        marker = 'x' if label in changed_labels else ('s' if i < len(features1) else 'o')
        plt.plot(pca_result[i, 0], pca_result[i, 1], marker=marker, color=color, markersize=8)
        plt.text(pca_result[i, 0], pca_result[i, 1], label, fontsize=8, color=color)
    plt.title('PCA of Image Features')
    custom_lines = [Line2D([0], [0], color=colors(i), lw=4) for i in range(len(unique_labels))]
    plt.legend(custom_lines, unique_labels)
    plt.show()

    # Plot t-SNE results
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(labels):
        idx = unique_labels.index(label)
        color = colors(idx)
        marker = 'x' if label in changed_labels else ('s' if i < len(features1) else 'o')
        plt.plot(tsne_result[i, 0], tsne_result[i, 1], marker=marker, color=color, markersize=8)
        plt.text(tsne_result[i, 0], tsne_result[i, 1], label, fontsize=8, color=color)
    plt.title('t-SNE of Image Features')
    custom_lines = [Line2D([0], [0], color=colors(i), lw=4) for i in range(len(unique_labels))]
    plt.legend(custom_lines, unique_labels)
    plt.show()


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


def compare_objects(image_path1, image_path2, csv_file_path):
    objects_info = read_objects_info(csv_file_path)
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)
    model = models.resnet18(pretrained=True)
    model.eval()
    # Remove the final layer to get the latent features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    features1 = []
    features2 = []
    labels = []
    changed_labels = []

    for object_info in objects_info:
        object_coords, object_name = object_info['coords'], object_info['name']
        cropped_image1 = crop_object(image1, object_coords)
        cropped_image2 = crop_object(image2, object_coords)
        image_tensor1 = preprocess_image(cropped_image1)
        image_tensor2 = preprocess_image(cropped_image2)
        feature1 = encode_image(model, image_tensor1).squeeze()
        feature2 = encode_image(model, image_tensor2).squeeze()
        features1.append(feature1)
        features2.append(feature2)
        labels.append(object_name)
        labels.append(object_name)

        similarity = torch.nn.functional.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0))
        print(f"{object_name} Cosine similarity: {similarity.item()}")

        if similarity.item() < 0.9:  # Assuming a similarity threshold
            print(f"Significant change detected in {object_name}.")
            changed_labels.append(object_name)

    features1 = torch.stack(features1)
    features2 = torch.stack(features2)

    # Visualize with PCA and t-SNE
    visualize_with_pca_tsne(features1, features2, labels, changed_labels)

    # Display images with highlighted changes if any
    changed_areas = [obj['coords'] for obj in objects_info if obj['name'] in changed_labels]
    if changed_areas:
        display_images_with_highlights(image_path1, image_path2, changed_areas)
    else:
        print("No significant changes detected between the images.")


# Example usage
image_path1 = 'out.png'
image_path2 = 'out (2).png'
csv_file_path = 'result.csv'
compare_objects(image_path1, image_path2, csv_file_path)
