import torch
import csv
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class ImageComparator:
    def __init__(self, model_checkpoint=None):
        # Load a pretrained model, either from a checkpoint or the default ResNet18
        if model_checkpoint:
            self.model = torch.load(model_checkpoint)
        else:
            self.model = models.resnet18(pretrained=True)
        self.model.eval()

    @staticmethod
    def load_image(image_path):
        # Load an image from the given path and convert it to RGB format
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except IOError:
            print(f"Error: Unable to load image at {image_path}. Check the file path and try again.")
            return None

    @staticmethod
    def crop_object(image, object_coords):
        # Crop a specified region from the image based on normalized coordinates
        img_width, img_height = image.size
        x_center, y_center, width, height = object_coords
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        left = x_center - (width / 2)
        top = y_center - (height / 2)
        right = x_center + (width / 2)
        bottom = y_center + (height / 2)
        return image.crop((left, top, right, bottom))

    @staticmethod
    def preprocess_image(image):
        # Preprocess the image to the format required by the model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

    def encode_image(self, image_tensor):
        # Encode the image using the model to get its feature representation
        with torch.no_grad():
            return self.model(image_tensor)

    @staticmethod
    def read_objects_info(csv_file_path):
        # Read object information from a CSV file
        objects_info = []
        try:
            with open(csv_file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    objects_info.append({
                        'coords': (float(row['X']), float(row['Y']), float(row['Width']), float(row['Height'])),
                        'name': row['Item Name']
                    })
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}. Please check the file path.")
        return objects_info

    def compare_objects(self, image_path1, image_path2, csv_file_path):
        # Compare objects between two images based on their features
        objects_info = self.read_objects_info(csv_file_path)
        image1 = self.load_image(image_path1)
        image2 = self.load_image(image_path2)
        if image1 and image2:
            changed_areas = []

            for object_info in objects_info:
                object_coords, object_name = object_info['coords'], object_info['name']
                cropped_image1 = self.crop_object(image1, object_coords)
                cropped_image2 = self.crop_object(image2, object_coords)
                image_tensor1 = self.preprocess_image(cropped_image1)
                image_tensor2 = self.preprocess_image(cropped_image2)
                feature1 = self.encode_image(image_tensor1)
                feature2 = self.encode_image(image_tensor2)
                similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
                print(f"{object_name} Cosine similarity: {similarity.item()}")

                if similarity.item() < 0.9:  # Assuming a similarity threshold
                    print(f"Significant change detected in {object_name}.")
                    changed_areas.append(object_coords)

            if changed_areas:
                self.display_images_with_highlights(image_path1, image_path2, changed_areas)
            else:
                print("No significant changes detected between the images.")

    @staticmethod
    def display_images_with_highlights(image_path1, image_path2, changed_areas):
        # Display the images side by side with highlighted areas where changes were detected
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        image1 = ImageComparator.load_image(image_path1)
        image2 = ImageComparator.load_image(image_path2)
        if image1 and image2:
            axs[0].imshow(image1)
            axs[1].imshow(image2)

            for area in changed_areas:
                img_width, img_height = image1.size
                x_center, y_center, width, height = area
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                left = x_center - (width / 2)
                top = y_center - (height / 2)
                rect1 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
                rect2 = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='r', facecolor='none')
                axs[0].add_patch(rect1)
                axs[1].add_patch(rect2)

            axs[0].axis('off')
            axs[1].axis('off')
            plt.show()
