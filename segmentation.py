import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from segment_anything import sam_model_registry, SamPredictor

class ObjectSegmenter:
    def __init__(self, image_path, csv_file_path, model_checkpoint):
        """
        Initialize the ObjectSegmenter.

        :param image_path: Path to the input image.
        :param csv_file_path: Path to the CSV file containing YOLO coordinates.
        :param model_checkpoint: Path to the SAM model checkpoint.
        """
        self.image_path = image_path
        self.csv_file_path = csv_file_path
        self.model_checkpoint = model_checkpoint
        self.img = self.load_image()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = self.load_model()

    def load_image(self):
        """
        Load the input image.

        :return: Loaded image in RGB format.
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError("Image not found at the specified path.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_model(self):
        """
        Load the SAM model.

        :return: Loaded SAM model.
        """
        sam = sam_model_registry["default"](checkpoint=self.model_checkpoint)
        return sam.to(device=self.device)

    def read_yolo_coordinates(self):
        """
        Read YOLO coordinates from the CSV file.

        :return: List of YOLO coordinates.
        """
        yolo_coords = []
        with open(self.csv_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                yolo_coords.append((float(row['X']), float(row['Y']), float(row['Width']), float(row['Height'])))
        return yolo_coords

    def yolo_to_absolute(self, yolo_coords):
        """
        Convert YOLO coordinates to absolute coordinates.

        :param yolo_coords: List of YOLO coordinates.
        :return: List of absolute coordinates.
        """
        h, w = self.img.shape[:2]
        abs_coords = []
        for x_center, y_center, width, height in yolo_coords:
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            abs_coords.append((x_min, y_min, int(width), int(height)))
        return abs_coords

    def segment_objects(self):
        """
        Segment objects in the image.

        :return: Filename of the segmented image.
        """
        predictor = SamPredictor(self.sam)
        predictor.set_image(self.img)
        yolo_coords = self.read_yolo_coordinates()
        abs_coords = self.yolo_to_absolute(yolo_coords)
        masks = []

        for x_min, y_min, width, height in abs_coords:
            center_x, center_y = x_min + width // 2, y_min + height // 2
            predicted_masks, scores, _ = predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            top_score = 0
            best_mask = None
            for score, mask in zip(scores, predicted_masks):
                if score > top_score:
                    top_score = score
                    best_mask = mask
            if best_mask is not None:
                masks.append(best_mask)

        final_image = self.apply_masks_to_image(masks)
        self.show_image(final_image)
        final_image_name='final_segment.png'
        cv2.imwrite(final_image_name, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        return final_image_name

    def apply_masks_to_image(self, masks):
        """
        Apply masks to the image.

        :param masks: List of masks.
        :return: Image with applied masks.
        """
        final_image = np.zeros_like(self.img)
        for mask in masks:
            object_mask = np.stack([mask] * 3, axis=-1)
            color = np.random.randint(0, 256, (3,), dtype=np.uint8)
            object_image = np.where(object_mask, color * mask[:, :, None], 0)
            final_image = np.where(object_mask, object_image, final_image)
        return final_image

    def show_image(self, img, title="Segmented Image"):
        """
        Display the image.

        :param img: Image to display.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    segmenter = ObjectSegmenter(
        image_path="/path/to/your/image.jpg",
        csv_file_path="/path/to/your/coordinates.csv",
        model_checkpoint="/path/to/your/model_checkpoint.pth"
    )
    segmenter.segment_objects()
