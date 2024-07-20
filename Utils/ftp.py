import os
import shutil
from pathlib import Path


def collect_and_rename_images(input_folders, output_folder):
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Iterate through each input folder
    for folder in input_folders:
        # Iterate through each file in the folder
        for filename in os.listdir(folder):
            # Check if the file is an image (you can customize the file extensions as needed)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Generate a new unique filename (you can customize the renaming logic)
                new_filename = f"image_{len(os.listdir(output_folder)) + 303}.png"

                # Build the paths for the source and destination
                source_path = os.path.join(folder, filename)
                destination_path = os.path.join(output_folder, new_filename)

                # Copy the file to the output folder with the new name
                shutil.copy(source_path, destination_path)
                print(f"Moved: {filename} to {new_filename}")


# Example usage:
# Specify the input folders and the output folder
source_folders = ['/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/bread', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/butter','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/carrot','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/chocolate','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/cucumber','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/cucumber1', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/egg','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/garlic', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/grape','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/grape1','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/honey', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/honey1', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/jam', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/juice1', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/ketchup', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/mayonnaise', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/milk','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/onion','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/peanut-butter', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/peppers','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/pickles','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/potato','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/Premixed-vinaigrettes', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/softdrink','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/softdrink1', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/tomatoes','/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/veggies', '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/yogurt']
destination_folder = '/Users/parthbhalodiya/Desktop/FridgeVision/untitled folder/'

collect_and_rename_images(source_folders, destination_folder)


