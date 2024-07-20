import os
from segmentation import ObjectSegmenter
from fridgevision import detect_objects
from latent_comparison import ImageComparator
from segment_masked_object import ObjectExtracter
from Recipe_LLM import read_ingredients_from_csv
from Recipe_LLM import generate_recipe
import cv2
import tempfile
import numpy as np
import requests
import csv
from PIL import Image



if __name__ == "__main__":
    url = "http://192.168.0.205:8080/shot.jpg"
    #model_path = "/Users/parthbhalodiya/Desktop/FridgeVision/sam/yolov8n.pt"
    model_path ="best.pt"
    csv_file_path = "object_detection_results.csv"

    # Step 1: Detect objects in the image
    output_frame = detect_objects(url, model_path, csv_file_path)
    temp_img_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    cv2.imwrite(temp_img_path, output_frame)

    # Step 2: Segment objects from the detected image
    segmenter = ObjectSegmenter(
        image_path=temp_img_path,
        csv_file_path=csv_file_path,
        model_checkpoint="/Users/parthbhalodiya/Desktop/FridgeVision/sam/sam_vit_h_4b8939.pth"
    )

    print("Performing object segmentation...")
    segmenter.segment_objects()

    # Step 3: Extract segmented objects with precomputed masks
    segment = ObjectExtracter(temp_img_path, segmenter.segment_objects())
    segment.segment_objects_with_precomputed_masks()

    # Step 4: Compare objects between two images
    comparator = ImageComparator()
    comparator.compare_objects(
        image_path1="/Users/parthbhalodiya/PycharmProjects/pythonProject/final/final_mask.png",
        image_path2=segmenter.segment_objects(),  # You need to specify the actual path or result here
        csv_file_path=csv_file_path
    )

    # Step 5: Replace and remove temporary files
    os.replace('final_extracted_segment.png', 'final_mask.png')
    os.remove(temp_img_path)

    # Step 6: Read ingredients from CSV and generate recipe
    ingredients = read_ingredients_from_csv(csv_file_path)
    print("FridgeVision's AI assistant is running...")
    generate_recipe(ingredients)