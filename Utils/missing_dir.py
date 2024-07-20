import os
import shutil

# Define the directories
dir1 = '/Users/parthbhalodiya/Downloads/FridgeVision.v1i.yolov8 2/train/labels'
dir2 = '/Users/parthbhalodiya/Downloads/untitled folder/FridgeVision.v1i.yolov8 3/train/labels/'
output_dir = '/Users/parthbhalodiya/Downloads/label'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of .txt files in both directories
files_in_dir1 = {file for file in os.listdir(dir1) if file.endswith('.txt')}
files_in_dir2 = {file for file in os.listdir(dir2) if file.endswith('.txt')}

# Find files that are in dir2 but not in dir1
missing_files = files_in_dir2 - files_in_dir1

# Copy missing files from dir2 to output_dir
for file_name in missing_files:
    source_path = os.path.join(dir2, file_name)
    destination_path = os.path.join(output_dir, file_name)
    shutil.copy2(source_path, destination_path)
    print(f"Copied '{file_name}' from '{dir2}' to '{output_dir}'")

print("Operation complete. All missing files have been copied.")
