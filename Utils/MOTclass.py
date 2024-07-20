import os
import os

def identify_class_id_changes(annotation_dirs, id_mapping):
    # Dictionary to store all class ID changes needed
    all_changes = {}

    for annotation_dir in annotation_dirs:
        # List all annotation files (.txt) in the specified directory
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]

        for filename in annotation_files:
            filepath = os.path.join(annotation_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            for line in lines:
                # Parse each line of the annotation file
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])  # Extract the class ID
                    if class_id in id_mapping:
                        new_class_id = id_mapping[class_id]
                        if new_class_id != class_id:
                            # Record the change needed
                            all_changes[class_id] = new_class_id

    return all_changes

def apply_class_id_changes(annotation_dirs, changes_to_apply):
    for annotation_dir in annotation_dirs:
        # List all annotation files (.txt) in the specified directory
        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]

        for filename in annotation_files:
            filepath = os.path.join(annotation_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            updated_lines = []
            for line in lines:
                # Parse each line of the annotation file
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])  # Extract the class ID
                    if class_id in changes_to_apply:
                        new_class_id = changes_to_apply[class_id]
                        parts[0] = str(new_class_id)  # Replace the old class ID with the new class ID
                    updated_line = ' '.join(parts) + '\n'
                    updated_lines.append(updated_line)

            # Write the updated annotation back to the file
            with open(filepath, 'w') as file:
                file.writelines(updated_lines)

# Example usage:
annotation_directories = [
    '/Users/parthbhalodiya/yolo/test/labels',
    '/Users/parthbhalodiya/yolo/train/labels',
    '/Users/parthbhalodiya/yolo/val/labels'
]
id_mapping = {
    52: 1,   # Change class ID 1 to 0
    53: 0,
    54: 38,
    55: 28,
    56: 5,
    57: 7,
    58: 39,
    59: 40,
    60: 41,
    61: 42

}
# Identify all necessary class ID changes
changes_to_apply = identify_class_id_changes(annotation_directories, id_mapping)

# Apply the identified changes
apply_class_id_changes(annotation_directories, changes_to_apply)

