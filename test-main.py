import os
from segmentation import ObjectSegmenter
from test import detect_objects
from latent_comparison import ImageComparator
from segment_masked_object import ObjectExtracter
from Recipe_LLM import read_ingredients_from_csv
from Recipe_LLM import generate_recipe
import cv2
import tempfile




if __name__ == "__main__":
    #url = "http://192.168.0.205:8080/shot.jpg"
    image_path = "/Users/parthbhalodiya/Downloads/44.jpeg"
    #model_path = "/Users/parthbhalodiya/Downloads/yolov8l.pt"
    model_path="/Users/parthbhalodiya/Desktop/FridgeVision/sam/yolov8n.pt"# Adjust the path as needed
    csv_file_path = "object_detection_results1.csv"
    output_image_path = "output8n.png"
    confidence_threshold = 0.3
    output_frame = detect_objects(image_path, model_path, csv_file_path, output_image_path, confidence_threshold)
    cv2.imshow("Object Detection", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    temp_img_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    cv2.imwrite(temp_img_path, output_frame)

    # Now, use the ObjectSegmenter class to perform object segmentation
''''
    segmenter = ObjectSegmenter(
        image_path=temp_img_path,
        csv_file_path=csv_file_path,
        model_checkpoint="/Users/parthbhalodiya/Desktop/FridgeVision/sam/sam_vit_h_4b8939.pth"
    )

    print("segmentation..")
    segmenter.segment_objects()

    segment = ObjectExtracter(temp_img_path, segmenter.segment_objects())
    segment.segment_objects_with_precomputed_masks()

    comparator = ImageComparator()
    comparator.compare_objects(
        image_path1="/Users/parthbhalodiya/PycharmProjects/pythonProject/final/final_mask.png",
        image_path2="final_extracted_segment.png",  # Specify the path to the second image
        csv_file_path=csv_file_path
    )
    os.replace('final_extracted_segment.png', 'final_mask.png')
    os.remove(temp_img_path)


    ingredients = read_ingredients_from_csv(csv_file_path)
    print("FridgeVision's AI assitant is running")
    generate_recipe(ingredients)'''''