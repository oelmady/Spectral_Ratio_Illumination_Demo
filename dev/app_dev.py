
import cv2
import numpy as np
import logging

class imgProcessor:
    def __init__(self, image_path: str, sr_map_path: str):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get image and map
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        self.sr_map = cv2.imread(sr_map_path, cv2.IMREAD_UNCHANGED)
        self.sr_map = self.sr_map /65535
        if self.sr_map is None:
            raise ValueError(f"Failed to load map from {sr_map_path}")

        # Attributes
        self.roi = None  # (x, y, w, h)
        self.roi_image = None

        self.logger.info(f"Initialized {self.__class__.__name__}")
        self.logger.info(f"Image loaded | Shape: {self.image.shape} | Max: {self.image.max()} | Min: {self.image.min()}")
        self.logger.info(f"Map loaded | Shape: {self.sr_map.shape} | Max: {self.sr_map.max()} | Min: {self.sr_map.min()}")
    
    def get_image(self) -> np.ndarray:

        return self.image
    
    def get_sr_map(self) -> np.ndarray:
    
        return self.sr_map
    
    def convert_img_to_log_space(self, linear_img) -> np.ndarray:
        """
        Converts a 16-bit linear image to log space, setting linear 0 values to 0 in log space.

        Parameters:
        -----------
        linear_img : np.array
            Input 16-bit image as a NumPy array.

        Returns:
        --------
        log_img : np.array
            Log-transformed image with 0 values preserved.
        """

        log_img = np.zeros_like(linear_img, dtype = np.float32)
        log_img[linear_img != 0] = np.log(linear_img[linear_img != 0])
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        return log_img

    def log_to_linear(self, log_img) -> None:
        """
        Converts log transofrmed image back to linear space in 16 bits.

        Parameters:
        -----------
        log_img : np.array
            Log-transformed image with values between 0 and 11.1.

        Returns:
        --------
        linear_img : np.array
            Visualization-ready 8-bit image.
        """

        linear_img = np.exp(log_img)
        return linear_img
    
    def convert_16bit_to_8bit(self, img) -> None:
        """
        Converts a 16-bit image to 8-bit by normalizing pixel values.

        Parameters:
        -----------
        img : np.array
            Input image array in 16-bit format (dtype: np.uint16).
        
        Returns:
        --------
        img_8bit : np.array
            Output image array converted to 8-bit (dtype: np.uint8).
        """
        img_normalized = np.clip(img / 255, 0, 255)
        img_8bit = np.uint8(img_normalized)
        return img_8bit

    def show_image(self, img, window_name=None, log=False):
        window_name = window_name if window_name else "image"
        if log:
            log_min, log_max = img.min(), img.max()
            log_normalized = 255 * (img - log_min) / (log_max - log_min + 1e-8)
            display_img = log_normalized.astype(np.uint8)
        else:
            display_img = img   
        cv2.imshow(window_name, display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def select_roi(self, img):
        self.logger.info("[Select ROI using mouse...")
        roi = cv2.selectROI("Select ROI", img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")

        if roi == (0, 0, 0, 0):
            self.logger.info("No ROI selected.")
            return

        x, y, w, h = roi
        roi_image = img[y:y+h, x:x+w]
        self.logger.info(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
        return roi_image, roi
    
    def shift_pixels_along_isd(self, log_image: np.ndarray, isd_map: np.ndarray, roi: tuple, distance: float) -> np.ndarray:
        """
        Shift pixels in log-RGB space along their ISD vector by a given distance within the ROI.
        
        Args:
            log_image (np.ndarray): Log-RGB image of shape (H, W, 3)
            isd_map (np.ndarray): ISD map of shape (H, W, 3), should be unit vectors
            roi (tuple): (x, y, w, h) rectangle specifying the ROI
            distance (float): Distance to shift pixels along ISD vector

        Returns:
            np.ndarray: Modified log-RGB image with ROI pixels shifted
        """
        shifted_image = log_image.copy()
        x, y, w, h = roi

        roi_pixels = log_image[y:y+h, x:x+w]
        roi_isd = isd_map[y:y+h, x:x+w]

        # Shift each pixel in the ROI
        shifted_roi = roi_pixels + distance * roi_isd

        # Replace back into image
        shifted_image[y:y+h, x:x+w] = shifted_roi
        return shifted_image, shifted_roi


    

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


    img_path = "/Users/mikey/Library/Mobile Documents/com~apple~CloudDocs/Code/code_projects/isd-annotator/annotations/images_for_testing/folder_6/images/maddineni_poojit_045.tif"
    sr_map_path = "/Users/mikey/Library/Mobile Documents/com~apple~CloudDocs/Code/code_projects/isd-annotator/annotations/images_for_testing/folder_6/isd_maps/maddineni_poojit_045_isd.tiff"
    img_processor = imgProcessor(img_path, sr_map_path)

    image = img_processor.get_image()
    # img_processor.show_image(image, "original image")

    sr_map = img_processor.get_sr_map()
    # img_processor.show_image(sr_map, "isd map")
    
    log_img = img_processor.convert_img_to_log_space(image)
    # img_processor.show_image(log_img, "log image", log=True)

    roi_image, roi = img_processor.select_roi(image)
    # img_processor.show_image(roi_image, "roi image", log=False)

    shifted_img, shifted_roi = img_processor.shift_pixels_along_isd(log_image=log_img, isd_map=sr_map, roi=roi, distance=1.0)
    img_processor.show_image(shifted_img, "new log image", log=True)

    new_img = img_processor.log_to_linear(shifted_img)
    logger.info(f"New linear image | Shape: {new_img.shape} | Max: {new_img.max()} | Min: {new_img.min()}")
    img_8bit = img_processor.convert_16bit_to_8bit(new_img)
    logger.info(f"New 8bit image | Shape: {img_8bit.shape} | Max: {img_8bit.max()} | Min: {img_8bit.min()}")
    img_processor.show_image(img_8bit, "new linear image")



if __name__ == "__main__":
    main()