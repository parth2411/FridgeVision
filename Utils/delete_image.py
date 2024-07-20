import os

# Path to the directory containing the image files
image_dir = '/Users/parthbhalodiya/yolo/train/images'

# Path to the directory containing the label files
label_dir = '/Users/parthbhalodiya/yolo/train/labels'

# Collect all label file names without extension
label_files = {os.path.splitext(file)[0] for file in os.listdir(label_dir) if file.endswith(".txt")}

# Process each image file in the directory
for filename in os.listdir(image_dir):
    # Assuming images are in .jpg format, adjust as necessary for other formats like .png
    if filename.endswith(".jpg"):
        image_base = os.path.splitext(filename)[0]

        # Check if there is a corresponding label file
        if image_base not in label_files:
            # No corresponding label file found, delete the image
            image_path = os.path.join(image_dir, filename)
            os.remove(image_path)
            print(f"Deleted: {image_path}")

print("Finished processing. Images without corresponding labels have been deleted.")