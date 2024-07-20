import os

# Set the path to your text files folder
text_folder_path = '/Users/parthbhalodiya/Desktop/FridgeVision/data/test-label'

# Set the path to the output folder
output_folder_path = '/Users/parthbhalodiya/Desktop/FridgeVision/data/test-label'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through all text files in the folder
for filename in os.listdir(text_folder_path):
    if filename.endswith('.txt'):
        text_file_path = os.path.join(text_folder_path, filename)

        # Read the content of the text file
        with open(text_file_path, 'r') as file:
            content = file.read()

        # Generate new filenames with prefixes
        new_filenames = [
            "blurred_" + filename,
            "manipulated_" + filename,
            "foggy_" + filename
        ]

        # Create copies of the text file with modified names
        for new_filename in new_filenames:
            new_file_path = os.path.join(output_folder_path, new_filename)

            # Write the content to the new file
            with open(new_file_path, 'w') as new_file:
                new_file.write(content)

print("Text file copies generated with modified names.")
