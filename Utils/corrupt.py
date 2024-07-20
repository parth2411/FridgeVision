import os
from PIL import Image

image_dir = '/Users/parthbhalodiya/Downloads/untitled folder/FridgeVision.v1i.yolov8 3/train/images'  # Update this path to your images directory

corrupt_files = []
for file_name in os.listdir(image_dir):
    if file_name.lower().endswith('.jpg'):
        file_path = os.path.join(image_dir, file_name)
        try:
            img = Image.open(file_path)  # Open the image file
            img.verify()  # Verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', file_path)  # Print out the names of corrupt files
            corrupt_files.append(file_path)

print("Identified corrupt JPEG files:", corrupt_files)