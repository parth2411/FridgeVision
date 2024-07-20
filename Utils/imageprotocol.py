from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you can customize the image file extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct the full file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open and resize the image
            with Image.open(input_path) as img:
                resized_img = img.resize(new_size)

                # Save the resized image to the output folder
                resized_img.save(output_path)

if __name__ == "__main__":
    # Specify the input folder, output folder, and the new size
    input_folder = "/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder"
    output_folder = "/Users/parthbhalodiya/Desktop/FridgeVision/dataset"
    new_size = (640, 640)  # Set the desired width and height

    # Resize images in the input folder and save to the output folder
    resize_images(input_folder, output_folder, new_size)
