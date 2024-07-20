
import cv2
import torch
from ultralytics import YOLO
import detect

def predict_objects_yolov5(image_path, model_path):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Read the image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Visualize the results on the image
    annotated_image = results.render()[0]

    return annotated_image

def predict_objects_yolov7(image_path, model_path):
    # Load the YOLOv7 model
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path=model_path)

    # Read the image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Visualize the results on the image
    annotated_image = results.render()[0]

    return annotated_image

def predict_objects_yolov8(image_path, model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Read the image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Visualize the results on the image
    annotated_image = results[0].plot()

    return annotated_image

def main():
    # Image path
    image_path = "/Users/parthbhalodiya/Downloads/11.jpeg"

    # Model paths
    yolov5_model_path = "/Users/parthbhalodiya/Downloads/yolov5n.pt"
    yolov7_model_path = "/Users/parthbhalodiya/Downloads/yolov7.pt"
    yolov8_model_path = "/Users/parthbhalodiya/Downloads/yolov8s.pt"

    # Predict objects using YOLOv5
    annotated_image_yolov5 = predict_objects_yolov5(image_path, yolov5_model_path)
    cv2.imshow("Object Detection - YOLOv5", annotated_image_yolov5)

    # Predict objects using YOLOv7
    annotated_image_yolov7 = predict_objects_yolov7(image_path, yolov7_model_path)
    cv2.imshow("Object Detection - YOLOv7", annotated_image_yolov7)

    # Predict objects using YOLOv8
    annotated_image_yolov8 = predict_objects_yolov8(image_path, yolov8_model_path)
    cv2.imshow("Object Detection - YOLOv8", annotated_image_yolov8)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()