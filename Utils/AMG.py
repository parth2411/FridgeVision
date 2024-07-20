import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from segment_anything import sam_model_registry, SamPredictor

def show_image(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def apply_masks_to_image(img, masks):
    final_image = np.zeros_like(img)
    for mask in masks:
        object_mask = np.stack([mask] * 3, axis=-1)
        color = np.random.randint(0, 256, (3,), dtype=np.uint8)
        object_image = np.where(object_mask, img, 0)
        final_image = np.where(object_mask, object_image, final_image)
    return final_image

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
sam = sam_model_registry["default"](checkpoint="/Users/parthbhalodiya/Desktop/FridgeVision/sam/sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

def yolo_to_absolute(img, yolo_coords):
    h, w = img.shape[:2]
    abs_coords = []
    for x_center, y_center, width, height in yolo_coords:
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        abs_coords.append((x_min, y_min, width, height))
    return abs_coords

def segment_objects(image_path, csv_file_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found at the specified path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    # Read YOLO coordinates from CSV
    yolo_coords = []
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yolo_coords.append((float(row['X']), float(row['Y']), float(row['Width']), float(row['Height'])))

    # Convert YOLO coordinates to absolute pixel coordinates
    abs_coords = yolo_to_absolute(img, yolo_coords)

    masks = []
    for x_min, y_min, width, height in abs_coords:
        center_x, center_y = x_min + width // 2, y_min + height // 2
        predicted_masks, scores, _ = predictor.predict(
            point_coords=np.array([[center_x, center_y]]),
            point_labels=np.array([1]),
            multimask_output=True
        )
        top_score = 0
        best_mask = None
        for score, mask in zip(scores, predicted_masks):
            if score > top_score:
                top_score = score
                best_mask = mask
        if best_mask is not None:
            masks.append(best_mask)

    final_image = apply_masks_to_image(img, masks)
    show_image(final_image)
    cv2.imwrite('final_image.png', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

segment_objects("/Users/parthbhalodiya/PycharmProjects/pythonProject/2.jpeg", "/Users/parthbhalodiya/PycharmProjects/pythonProject/result.csv")
