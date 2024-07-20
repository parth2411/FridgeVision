import cv2
import numpy as np
import requests
import csv
from ultralytics import YOLO
from PIL import Image
import math
from segmentation import ObjectSegmenter

#classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "milk", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
classNames = ['apple', 'banana', 'beef', 'blueberries', 'bread', 'broccoli', 'butter', 'carrot', 'cheese', 'chicken', 'chocolate', 'corn', 'eggs', 'flour', 'goat_cheese', 'green_beans', 'ground_beef', 'ham', 'heavy_cream', 'lemon', 'mayonaise', 'milk', 'mushrooms', 'natural_yoghurt', 'onion', 'orange', 'pepper', 'potato', 'shrimp', 'spinach', 'strawberries', 'sugar', 'sweet_potato', 'tomato']

# Adjust brightness and contrast of the image
def resize_image(image, width, height, interpolation=cv2.INTER_LINEAR):
    resized_image = cv2.resize(image, (width, height), interpolation=interpolation)
    return resized_image
def adjust_brightness_contrast(image, brightness=30, contrast=30):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image

def enhance_image(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Further enhancements could be added here if desired.
    image_enhanced = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    return image_enhanced

def get_image_from_ip_camera(url):
    response = requests.get(url)
    image_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, -1)
    return image

def detect_objects(url, model_path, csv_file_path):
    # Load the YOLO model
    model = YOLO(model_path)

    image = get_image_from_ip_camera(url)
    frame = resize_image(image,640,640)
    #frame = adjust_brightness_contrast(frame)
    #frame = enhance_image(frame)
    results = model(frame, stream=True)

    # Save results to CSV
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Item Name', 'X', 'Y', 'Width', 'Height', 'Confidence'])

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list

            # Normalize coordinates
            height, width, _ = frame.shape
            x1_norm = x1 / width
            y1_norm = y1 / height
            x2_norm = x2 / width
            y2_norm = y2 / height

            print("Normalized Coordinates --->", [x1_norm, y1_norm, x2_norm, y2_norm])
            # put box in cam
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                class_name = classNames[cls]
                x_norm = (x1_norm + x2_norm) / 2  # Normalize center x-coordinate
                y_norm = (y1_norm + y2_norm) / 2  # Normalize center y-coordinate
                w_norm = x2_norm - x1_norm  # Normalize width
                h_norm = y2_norm - y1_norm  # Normalize height
                confidence = confidence

                writer.writerow([class_name, x_norm, y_norm, w_norm, h_norm, confidence])

    print(f"Object detection results saved to {csv_file_path}")
    return frame

if __name__ == "__main__":
    # URL for the IP camera
    url = "http://192.168.199.157:8080/shot.jpg"
    model_path = "/Users/parthbhalodiya/Desktop/FridgeVision/sam/yolov8n.pt"  # Adjust the path as needed
    csv_file_path = "object_detection_results.csv"

    output_frame = detect_objects(url, model_path, csv_file_path)
    cv2.imshow("Object Detection", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()