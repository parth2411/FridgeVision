import cv2
import numpy as np
import matplotlib.pyplot as plt


class ObjectExtracter:
    def __init__(self, image_path, mask_path):
        """
        Initialize the ObjectExtracter with paths to the image and mask.

        :param image_path: Path to the input image
        :param mask_path: Path to the mask image
        """
        self.image_path = image_path
        self.mask_path = mask_path

    @staticmethod
    def show_image(img):
        """
        Display an image using matplotlib.

        :param img: The image to display
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    @staticmethod
    def apply_masks_to_image(img, mask):
        """
        Apply masks to the image. Each unique value in the mask is considered a separate object.

        :param img: The original image
        :param mask: The mask image
        :return: The image with applied masks
        """
        unique_masks = np.unique(mask)
        final_image = np.zeros_like(img)
        for unique_mask in unique_masks:
            if unique_mask == 0:  # Skip background
                continue
            object_mask = (mask == unique_mask)
            color = np.random.randint(0, 256, (3,), dtype=np.uint8)
            object_image = np.where(object_mask[..., None], img, 0)  # Apply color to the mask
            final_image = np.where(object_mask[..., None], object_image, final_image)
        return final_image

    def segment_objects_with_precomputed_masks(self):
        """
        Segment objects in the image using precomputed masks, display and save the final image.

        :return: The filename of the final saved image
        """
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at the specified path: {self.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load the mask image
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask image not found at the specified path: {self.mask_path}")

        final_image = self.apply_masks_to_image(img, mask)
        self.show_image(final_image)
        final_image_name = 'final_extracted_segment.png'
        cv2.imwrite(final_image_name, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        return final_image_name



# Uncomment and modify the paths to use the ObjectExtracter class
# segmenter = ObjectExtracter("/path/to/your/image.jpg", "/path/to/your/mask.png")
# segmenter.segment_objects_with_precomputed_masks()
