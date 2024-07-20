import os

# Define the directories for the images and labels
image_dir = '/Users/parthbhalodiya/yolo/val/images'
label_dir = '/Users/parthbhalodiya/yolo/val/labels'

# Helper function to delete files based on a pattern
def delete_files(directory, pattern):
    for filename in os.listdir(directory):
        if pattern in filename:
            file_path = os.path.join(directory, filename)
            os.remove(file_path)
            print(f"Deleted {file_path}")

# Delete augmented images and labels
delete_files(image_dir, '_aug_')
delete_files(label_dir, '_aug_')

print("Cleanup complete.")
