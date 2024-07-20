import os

# Set the path to the directory containing the label files
label_dir = '/Users/parthbhalodiya/yolo/train/labels'

# Set the list of class indices you want to keep
classes_to_keep = {52, 53, 54, 55, 56, 57, 58, 59, 60, 61}  # Example: keeping classes 0-9

# Process each file in the directory
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        path = os.path.join(label_dir, filename)
        with open(path, 'r') as file:
            lines = file.readlines()

        # Keep only the lines that start with a class index in classes_to_keep
        new_lines = [line for line in lines if int(line.split()[0]) in classes_to_keep]

        # Write the filtered lines back to the file
        with open(path, 'w') as file:
            file.writelines(new_lines)
            # Optional: Delete the file if it becomes empty after filtering
            if not new_lines:
                os.remove(path)

print("Finished processing label files to keep specified classes.")
