FridgeVision is an innovative food management system designed to assist individuals with dementia in maintaining proper nutrition and independent living. By leveraging state-of-the-art computer vision techniques, such as object detection using YOLOv8, segmentation with SAM, and latent space analysis, FridgeVision accurately identifies, organizes, and monitors food items within the refrigerator. 

**YouTube**
Demo video: https://youtu.be/yLCwcd_7rjM

**Features**

Object detection: Utilizes YOLOv8 to accurately detect and localize food items within the refrigerator. Object segmentation: Employs the Segment
Anything Model (SAM) to precisely segment detected food items at the pixel level. 
Latent space analysis: Compares latent representations of food items using ResNet18 to track changes in refrigerator contents over time. 
Recipe recommendation: Generates personalized recipe suggestions based on the available ingredients using the Llama3 language model.
User-friendly interface: Provides an intuitive and accessible user interface for individuals with dementia and their caregivers.

**Prerequisites**

Python 3.7 or higher 
PyTorch 
OpenCV 
Matplotlib 
Pillow 
NumPy 
requests 
csv
ollama

**Installation**

Install the required dependencies: pip install -r requirements.txt

**Usage**

Update the configuration variables in the main.py file, such as the URL
for the IP camera, file paths for the YOLOv8 model, SAM model
checkpoint, and CSV file. Run the main.py script: 
python main.py

The script will perform the following steps:
Detect objects in the image using YOLOv8 Segment objects from the
detected image using SAM Extract segmented objects with precomputed
masks Compare objects between two images using latent space analysis
Generate a personalized recipe based on the detected ingredients using
Llama3

View the generated results, including the segmented image, highlighted
changes, and the recommended recipe.

**Directory Structure**

main.py: The main script that orchestrates the entire FridgeVision pipeline. 
fridgevision.py: Contains the object detection code using YOLOv8. 
segmentation.py: Implements the object segmentation functionality using SAM. 
segment_masked_object.py: Extracts segmented objects with precomputed masks. 
latent_comparison.py: Performs latent space analysis to compare objects between images. 
Recipe_LLM.py: Generates personalized recipe recommendations using the Llama3 language model. 
requirements.txt: Lists the required Python dependencies.



Contact For any inquiries or questions, please contact
parthbhalodiya24@gmail.com.

**YoloV8**
download link: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

**SAM**
download link: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

**Download Dataset and yolov8 training folder**
option 1 link: https://app.roboflow.com/object-detection-mybii/fridgevision/browse
option 2 Download Dataset from here : https://drive.google.com/drive/folders/1Fx2JmGw5udmQyt2kuuSK0BuN4JBgxo6B?usp=sharing
model: https://app.roboflow.com/object-detection-mybii/fridgevision/3
yolo training folder: https://drive.google.com/drive/folders/12pxr0pWuLDVN9-w4KtjsTVhYiPVpw0yI?usp=sharing

**utils**
utils folder contains all utility file like augmentation file, mot file, FTP, split, other operational files, etc.

This README file provides an overview of the FridgeVision project, its
features, installation instructions, usage guidelines, directory
structure, and acknowledgements. 
