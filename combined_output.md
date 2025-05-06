```src\circuit_analyzer.py
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from .utills import (
    load_classes,
    non_max_suppression_by_area,
)

# +++ SAM 2 Imports +++
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
# Import SAM2 components from sam2_infer
from .sam2_infer import (
    SAM2Transforms,
    get_modified_sam2,
    device
)

class CircuitAnalyzer():
    def __init__(self, yolo_path='D:/SDP_demo/models/YOLO/best_large_model_yolo.pt', 
                 sam2_config_path='D:/SDP_demo/models/configs/sam2.1_hiera_l.yaml',
                 sam2_base_checkpoint_path='D:/SDP_demo/models/SAM2/sam2.1_hiera_large.pt',
                 sam2_finetuned_checkpoint_path='D:/SDP_demo/models/SAM2/best_miou_model_SAM_latest.pth',
                 use_sam2=True,
                 debug=False):
        self.yolo = YOLO(yolo_path)
        self.debug = debug
        self.classes = load_classes()
        self.classes_names = set(self.classes.keys())
        self.non_components = set(['text', 'junction', 'crossover', 'terminal', 'vss', 'explanatory', 'circuit', 'vss'])
        self.source_components = set(['voltage.ac', 'voltage.dc', 'voltage.battery', 'voltage.dependent', 'current.dc', 'current.dependent'])
        
        problematic = set(['__background__', 'inductor.coupled', 'operational_amplifier.schmitt_trigger', 'mechanical', 'optical', 'block', 'magnetic', 'antenna', 'relay'])
        reducing = set(['operational_amplifier.schmitt_trigger', 'integrated_circuit.ne555', 'resistor.photo', 'diode.thyrector'])
        deleting = set(['optical', '__background__', 'inductor.coupled', 'mechanical', 'block','magnetic'])
        unknown = set(['relay','antenna','diac','triac', 'crystal','antenna', 'probe', 'probe.current', 'probe.voltage', 'optocoupler', 'socket', 'fuse', 'speaker', 'motor', 'lamp', 'microphone','transistor.photo','xor', 'and', 'or', 'not', 'nand', 'nor'])
        
        self.classes_names = self.classes_names - deleting - unknown - reducing
        self.classes = {key: value for key, value in self.classes.items() if key in self.classes_names}
        self.classes = {key: i for i, key, in enumerate(self.classes.keys())}
        
        self.project_classes = set(['gnd', 'voltage.ac', 'voltage.dc', 'voltage.battery', 'resistor', 'voltage.dependent', 'current.dc', 'current.dependent', 'capacitor', 'inductor', 'diode'])
        self.netlist_map = {
            'resistor': 'R',
            'resistor.adjustable': 'R',
            'capacitor': 'C',
            'capacitor.unpolarized': 'C',
            'capacitor.polarized': 'C',
            'capacitor.adjustable': 'C',
            'inductor': 'L',
            'inductor.ferrite': 'L',
            'diode': 'D',
            'diode.light_emitting': 'D',
            'diode.zener': 'D',
            'transistor.bjt': 'Q',
            'transistor.fet': 'M',
            'voltage.ac': 'V',
            'voltage.dc': 'V',
            'voltage.battery': 'V',
            'voltage.dependent': 'E',
            'current.dc': 'I',
            'current.dependent': 'G',
            'vss': 'GND',
            'gnd': '0',
            'switch': 'S',
            'integrated_circuit': 'X',
            'integrated_circuit.voltage_regulator': 'X',
            'operational_amplifier': 'X',
            'thyristor': 'Q',
            'transformer': 'T',
            'varistor': 'RV',
            'terminal': 'N',
            'junction': '',
            'crossover': '',
            'explanatory': '',
            'text': '',
            'unknown': 'UN',
        }
        
        # +++ SAM2 Initialization +++
        self.use_sam2 = use_sam2
        self.sam2_model = None
        self.sam2_transforms = None
        self.sam2_device = None

        if self.use_sam2:
            try:
                print("--- Initializing SAM2 ---")
                # Import SAM2 model and transforms from sam2_infer
                # We use the already initialized model from sam2_infer
                from .sam2_infer import modified_sam2, _transforms, device
                
                self.sam2_device = device
                print(f"Using SAM2 device: {self.sam2_device}")
                
                # Use the pre-initialized model
                self.sam2_model = modified_sam2
                self.sam2_model.eval()  # Ensure it's in evaluation mode
                
                # Use the pre-initialized transforms
                self.sam2_transforms = _transforms
                
                print("SAM2 Model imported successfully")
                print("--- SAM2 Ready ---")

            except Exception as e:
                print(f"An unexpected error occurred during SAM2 initialization: {e}")
                import traceback
                print(traceback.format_exc())
                print("SAM2 will be disabled.")
                self.use_sam2 = False
                self.sam2_model = None
                self.sam2_transforms = None
                self.sam2_device = None
        else:
            print("SAM2 usage is disabled in configuration.")
        
    def bboxes(self, image):
        results = self.yolo.predict(image, verbose=True)
        results = results[0]
        img_classes = results.boxes.cls.cpu().numpy().tolist()
        img_classes = [results.names[int(num)] for num in img_classes]
        confidences = results.boxes.conf.cpu().numpy().tolist()
        boxes = results.boxes.xyxy.cpu().numpy().tolist()
        boxes = [{'class': img_classes[i], 'confidence': confidences[i], 'xmin': round(xmin), 'ymin': round(ymin), 'xmax': round(xmax), 'ymax': round(ymax), 'id': f"{img_classes[i]}_{xmin}_{round(ymin)}_{round(xmax)}_{round(ymax)}"} for i, (xmin, ymin, xmax, ymax) in enumerate(boxes)]
        return boxes

    def enhance_lines(self, image):
        """
        Enhances lines in an image using morphological operations.

        Args:
            image_path (str): The path to the input image.

        Returns:
            numpy.ndarray: The processed image.
        """

        # Load the image
        img = image
        # Apply Gaussian blur for initial noise reduction
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        # Define the kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        # Apply dilation to connect broken lines and thicken them
        dilated = cv2.dilate(blurred, kernel, iterations=2)
        # Apply erosion to refine the line thickness and remove small noise
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(dilated, kernel, iterations=2)
        return eroded
    
    def segment_circuit(self, img):
        # converting to grey scale
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # applying adaptive threshold
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 21)
        return img
    
    def segment_with_sam2(self, image_np_bgr):
        """
        Segments the circuit diagram using the loaded SAM 2 model.

        Args:
            image_np_bgr (np.ndarray): Input image in BGR format (from cv2.imread).

        Returns:
            np.ndarray: Binary mask (uint8, 0 or 255) where 255 represents wires/connections,
                        at the original image resolution. Returns None if SAM 2 is not available or fails.
        """
        if not self.use_sam2 or self.sam2_model is None or self.sam2_transforms is None:
            print("SAM 2 is not available or not initialized. Cannot segment.")
            return None

        print("Segmenting with SAM 2...")
        try:
            # 1. Prepare Input Image
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_np_rgb)
            orig_hw = image_np_rgb.shape[:2]

            # 2. Preprocess
            image_tensor = self.sam2_transforms(image_pil).unsqueeze(0).to(self.sam2_device)

            # 3. Inference
            self.sam2_model.eval() # Ensure evaluation mode
            with torch.no_grad():
                # Use the SAM2 model from sam2_infer
                high_res_mask, low_res_mask, _ = self.sam2_model(image_tensor)
                # Choose high_res mask for better detail
                chosen_mask = high_res_mask

            # 4. Postprocess
            final_mask_tensor = self.sam2_transforms.postprocess_masks(chosen_mask, orig_hw) # Shape (1, 1, H, W)

            # 5. Convert to Binary NumPy Mask
            mask_squeezed = final_mask_tensor.detach().cpu().squeeze() # Shape (H, W)
            mask_binary_np = (mask_squeezed > 0.0).numpy().astype(np.uint8) * 255 # Threshold and scale to 0/255

            print("SAM 2 Segmentation successful.")
            return mask_binary_np

        except Exception as e:
            print(f"Error during SAM 2 segmentation: {e}")
            import traceback
            print(traceback.format_exc())
            return None # Return None on failure
    
    def get_contours(self, img, area_threshold=0.00040):
        """
        Finds contours of black lines on a white background and visualizes them individually.

        Args:
            image_path (str): The path to the input image.
        """

        # Load the image
        # Invert the image if necessary (lines should be white, background black)
        if cv2.mean(img)[0] > 127:  # Check if the image is mostly white
            img = 255 - img 

        # Threshold the image to create a binary mask (black lines on white background)
        img[img==255] = 1
        # Find contours 
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank color image for visualization
        contour_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        normalizer = (img.shape[0]*img.shape[1])
        contours = [x for x in contours if cv2.contourArea(x)/normalizer > area_threshold]

        contours = [{'id': i, 'contour': contour, 'area': cv2.contourArea(contour)/normalizer, 'rectangle': cv2.boundingRect(contour)} for i, contour in enumerate(contours)]

        # Iterate through the contours and visualize each
        for contour in contours:
            # Choose a random color for each contour
            color = (np.random.randint(0, 255), 
                     np.random.randint(0, 255), 
                     np.random.randint(0, 255))
            cv2.drawContours(contour_img, [contour['contour']], -1, color, 2)


            font=cv2.FONT_HERSHEY_SIMPLEX
            font_scale=0.5
            color=(255, 0, 0) 
            thickness=2

            # Calculate the contour's centroid
            M = cv2.moments(contour['contour'])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0 

            # Get the text for the current contour
            text = str(contour['id'])

            # Put the text on the image
            cv2.putText(contour_img, text, (cX + 10, cY + 10), 
                        font, font_scale, color, thickness)
        return contours, contour_img

    def resize_bboxes(self, bboxes, width_scale, height_scale):
        """
        Resizes bounding boxes using scaling factors.

        Args:
            bboxes (list): A list of bounding boxes, where each bounding box 
                           is a tuple (xmin, ymin, xmax, ymax).
            width_scale (float): The factor by which the width was scaled.
            height_scale (float): The factor by which the height was scaled.

        Returns:
            list: A list of resized bounding boxes.
        """
        resized_bboxes = []
        for bbox in bboxes:
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']

            xmin_resized = int(xmin * width_scale)
            ymin_resized = int(ymin * height_scale)
            xmax_resized = int(xmax * width_scale)
            ymax_resized = int(ymax * height_scale)

            resized_bboxes.append({'class': bbox['class'], 'xmin': xmin_resized, 'xmax': xmax_resized, 'ymin': ymin_resized, 'ymax': ymax_resized,})
        return resized_bboxes
    
    def enumerate_components(self, image, bboxes = None, excluded_labels=None):
        # Read the image using a helper function
        image = image.copy()
        if excluded_labels is None:
            excluded_labels = self.non_components

        # Get the image dimensions
        image_height, image_width = image.shape[:2]

        # Calculate a relative font scale based on the image height
        font_scale = max(0.5, image_height / 500.0)
        thickness = int(max(1, image_height / 400.0))

        bbox_counter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)  # Red color for the text
        
        if bboxes == None:
            bboxes = self.bboxes(image)
            bboxes = non_max_suppression_by_area(bboxes, iou_threshold=0.6)

        # List to store bounding boxes of components that represent text or other objects
        text_bboxes = []

        # Convert image to grayscale for background intensity checking
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        def calculate_overlap_area(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2

            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)

            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)

            return inter_width * inter_height

        def calculate_background_score(x, y, width, height):
            """Calculate how suitable the background is for text placement.
            Returns a score where higher values mean better placement."""
            if (y < 0 or x < 0 or 
                y + height > image_height or 
                x + width > image_width):
                return float('-inf')
            
            # Extract the region where text would be placed
            region = gray_image[y:y + height, x:x + width]
            
            # Calculate mean intensity of the region
            mean_intensity = np.mean(region)
            
            # Calculate standard deviation of intensity (for contrast)
            std_intensity = np.std(region)
            
            # Prefer lighter backgrounds (higher intensity) with less variation
            # Higher score means better placement
            return mean_intensity - std_intensity * 0.5

        def total_overlap_for_position(x, y, w, h, text_bboxes):
            total_overlap = 0
            for text_bbox in text_bboxes:
                overlap_area = calculate_overlap_area((x, y, x + w, y + h), text_bbox)
                total_overlap += overlap_area
            return total_overlap

        def find_optimal_position(xmin, ymin, xmax, ymax, text_width, text_height):
            # Define candidate positions around the bounding box
            positions = []
            
            # Center positions
            positions.extend([
                (xmin + (xmax - xmin) // 2 - text_width // 2, ymin - text_height - 5),  # Above center
                (xmin + (xmax - xmin) // 2 - text_width // 2, ymax + 5),                # Below center
                (xmin - text_width - 5, ymin + (ymax - ymin) // 2 - text_height // 2),  # Left center
                (xmax + 5, ymin + (ymax - ymin) // 2 - text_height // 2)               # Right center
            ])
            
            # Corner positions
            positions.extend([
                (xmin - text_width - 5, ymin - text_height - 5),  # Top left
                (xmax + 5, ymin - text_height - 5),               # Top right
                (xmin - text_width - 5, ymax + 5),                # Bottom left
                (xmax + 5, ymax + 5)                             # Bottom right
            ])
            
            # Inside positions (if bounding box is large enough)
            if (xmax - xmin) > text_width * 1.2 and (ymax - ymin) > text_height * 1.2:
                positions.extend([
                    (xmin + 5, ymin + 5),                        # Inside top left
                    (xmax - text_width - 5, ymin + 5),           # Inside top right
                    (xmin + 5, ymax - text_height - 5),          # Inside bottom left
                    (xmax - text_width - 5, ymax - text_height - 5)  # Inside bottom right
                ])

            # Add additional positions in a grid pattern around the bounding box
            grid_step = max(text_height, text_width)
            for dx in range(-grid_step*2, grid_step*2 + 1, grid_step):
                for dy in range(-grid_step*2, grid_step*2 + 1, grid_step):
                    positions.append((
                        xmin + (xmax - xmin) // 2 - text_width // 2 + dx,
                        ymin + (ymax - ymin) // 2 - text_height // 2 + dy
                    ))

            best_pos = None
            best_score = float('-inf')
            
            # Try all positions and find the optimal one considering both overlap and background
            for pos_x, pos_y in positions:
                # Check if position keeps text within image bounds
                if (0 <= pos_x <= image_width - text_width and 
                    0 <= pos_y <= image_height - text_height):
                    
                    # Calculate background score
                    bg_score = calculate_background_score(pos_x, pos_y, text_width, text_height)
                    
                    # Calculate overlap penalty (negative because we want to minimize overlap)
                    overlap_penalty = -total_overlap_for_position(
                        pos_x, pos_y, text_width, text_height, text_bboxes
                    ) * 0.001  # Scale factor to balance with background score
                    
                    # Combined score (higher is better)
                    total_score = bg_score + overlap_penalty
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_pos = (pos_x, pos_y)
            
            # If no valid position found, force it inside the bounding box at the lightest spot
            if best_pos is None:
                fallback_positions = [
                    (min(max(xmin, 0), image_width - text_width),
                     min(max(ymin, 0), image_height - text_height)),
                    (min(xmin + (xmax - xmin) // 2 - text_width // 2, image_width - text_width),
                     min(ymin + (ymax - ymin) // 2 - text_height // 2, image_height - text_height))
                ]
                
                best_fallback_score = float('-inf')
                for pos_x, pos_y in fallback_positions:
                    score = calculate_background_score(pos_x, pos_y, text_width, text_height)
                    if score > best_fallback_score:
                        best_fallback_score = score
                        best_pos = (pos_x, pos_y)
            
            return best_pos
        
        bbox_ids = []
        # Loop through the bounding boxes
        for bbox in bboxes:
            c = bbox['class']
            if excluded_labels and c in excluded_labels:
                text_bboxes.append((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
                continue

            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])

            bbox_counter += 1
            
            # Get text size
            text_size = cv2.getTextSize(f"{bbox_counter}", font, font_scale, thickness)[0]
            text_width, text_height = text_size

            # Find optimal position for the text
            pos_x, pos_y = find_optimal_position(xmin, ymin, xmax, ymax, text_width, text_height)
            
            # Add text height to y-position since OpenCV draws text above the y-coordinate
            pos_y += text_height
            
            # Draw the text with a white outline for better visibility on any background
            # Draw outline
            cv2.putText(image, f"{bbox_counter}", (pos_x, pos_y), font, 
                       font_scale, (255, 255, 255), thickness + 2)
            # Draw main text
            cv2.putText(image, f"{bbox_counter}", (pos_x, pos_y), font, 
                       font_scale, color, thickness)
            bbox_id = deepcopy(bbox)
            bbox_id['id'] = bbox_counter
            bbox_ids.append(bbox_id)
            
        # Convert image from BGR to RGB for displaying with Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            plt.axis("off")
            plt.show()
            
        return image, bbox_ids


    def resize_image_keep_aspect(self, image, bboxes, new_height=600):
        """
        Resizes an image to a specific height while maintaining aspect ratio.

        Args:
            image: The input image (NumPy array).
            new_height: The desired height of the resized image.

        Returns:
            The resized image (NumPy array).
        """

        height, width = image.shape[:2]
        aspect_ratio = width / height

        # Calculate the new width based on the aspect ratio
        new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))
        resized_bboxes = self.resize_bboxes(bboxes, new_width/width, new_height/height)

        return resized_image, resized_bboxes
    
    def is_point_near_bbox(self, point, bbox, pixel_threshold):
        """
        Checks if a point is close enough to a bounding box within a given 
        pixel threshold, considering points inside the box as near.

        Args:
            point (tuple): The (x, y) coordinates of the point.
            bbox (tuple): The bounding box coordinates (xmin, ymin, xmax, ymax).
            pixel_threshold (int): The maximum allowed distance in pixels 
                                    for the point to be considered near the bbox.

        Returns:
            bool: True if the point is near the bounding box, False otherwise.
        """

        px, py = point
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']

        # 1. Check if the point is inside the bounding box
        if xmin <= px <= xmax and ymin <= py <= ymax:
            return True  # Point is inside, so it's definitely near

        # 2. If not inside, calculate distances to edges and apply threshold
        dist_left = abs(px - xmin)
        dist_right = abs(px - xmax)
        dist_top = abs(py - ymin)
        dist_bottom = abs(py - ymax)

        if (dist_left <= pixel_threshold or dist_right <= pixel_threshold or
            dist_top <= pixel_threshold or dist_bottom <= pixel_threshold):
            return True
        else:
            return False
    
    def get_emptied_mask(self, image, bboxes):
        mask = self.segment_circuit(image)
        if self.debug:
            self.show_image(mask, 'Segmented')
        # Check if background is predominantly black
        # Calculate the percentage of black pixels in the border regions
        
        image_copy = mask.copy()  # Work with a copy of the potentially inverted mask
        
        for bbox in bboxes:
            # If the class is not 'crossover', 'junction', or 'terminal', zero out the bbox region
            if bbox['class'] not in ('crossover', 'junction', 'terminal', 'circuit', 'vss'):
                image_copy[int(bbox['ymin']):int(bbox['ymax']), 
                          int(bbox['xmin']):int(bbox['xmax'])] = 0
                
            # If the class is 'circuit', keep only the pixels within the bounding box
            if bbox['class'] == 'circuit':
                new_image_copy = np.zeros_like(image_copy)
                new_image_copy[int(bbox['ymin']):int(bbox['ymax']), 
                             int(bbox['xmin']):int(bbox['xmax'])] = \
                    image_copy[int(bbox['ymin']):int(bbox['ymax']), 
                              int(bbox['xmin']):int(bbox['xmax'])]
                image_copy = new_image_copy
        
        return image_copy
    
    def show_image(self, img, title="Image"):
        plt.figure(figsize=(10, 10))
        plt.imshow(img) # 'gray' colormap for grayscale images
        plt.title(title)
        plt.axis('off') # Hide axes ticks and labels
        plt.show()
        return img
    
    
    def get_node_connections(self, image, bboxes, original_size=False):
        original = image.copy()
        image = image.copy()
        
        # --- Segmentation ---
        if self.use_sam2:
            # Try SAM2 segmentation
            wire_mask = self.segment_with_sam2(image)
            if wire_mask is None:
                print("SAM2 segmentation failed, falling back to traditional segmentation")
                # Fallback to traditional segmentation
                emptied_mask, resized_bboxes = self.resize_image_keep_aspect(self.get_emptied_mask(image, bboxes), bboxes)
            else:
                # Get emptied mask from SAM2 segmentation
                emptied_mask = wire_mask.copy()
                height, width = emptied_mask.shape[:2]
                
                # Empty out component areas
                for bbox in bboxes:
                    if bbox['class'] not in ('crossover', 'junction', 'terminal', 'circuit', 'vss'):
                        ymin, ymax = max(0, int(bbox['ymin'])), min(height, int(bbox['ymax']))
                        xmin, xmax = max(0, int(bbox['xmin'])), min(width, int(bbox['xmax']))
                        if ymin < ymax and xmin < xmax:
                            emptied_mask[ymin:ymax, xmin:xmax] = 0
                
                # Apply circuit bbox if present
                for bbox in bboxes:
                    if bbox['class'] == 'circuit':
                        new_mask = np.zeros_like(emptied_mask)
                        ymin, ymax = max(0, int(bbox['ymin'])), min(height, int(bbox['ymax']))
                        xmin, xmax = max(0, int(bbox['xmin'])), min(width, int(bbox['xmax']))
                        if ymin < ymax and xmin < xmax:
                            new_mask[ymin:ymax, xmin:xmax] = emptied_mask[ymin:ymax, xmin:xmax]
                        emptied_mask = new_mask
                        break
                
                # Resize for processing
                emptied_mask, resized_bboxes = self.resize_image_keep_aspect(emptied_mask, bboxes)
        else:
            # Use traditional segmentation
            emptied_mask, resized_bboxes = self.resize_image_keep_aspect(self.get_emptied_mask(image, bboxes), bboxes)
        
        if self.debug:
            self.show_image(emptied_mask, 'Emptied')
            
        enhanced = self.enhance_lines(emptied_mask)
        if self.debug:
            self.show_image(enhanced, 'Enhanced')
            
        contours, contour_image = self.get_contours(enhanced)
        if self.debug:
            self.show_image(contour_image, 'Contours')
            
        nodes = {i['id']:{'id': i['id'], 'components': [], 'contour': i['contour']} for i in contours}

        harris_image = contour_image.copy()
        for contour in contours:
            # Convert the contour to a grayscale image (for Harris corner detection)
            mask = np.zeros_like(enhanced, dtype=np.uint8)
            cv2.drawContours(mask, [contour['contour']], -1, (255, 255, 255), -1)
            normalizer = mask.shape[0] * mask.shape[1]
            # Harris corner detection
            gray = mask
            gray = 255 - gray
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.015)
            dst = cv2.dilate(dst, None)  # Dilate to mark corners stronger
            # Threshold for corner detection
            corners = dst > 0.1 * dst.max()
            corner_points = np.argwhere(corners)
            thresh = 0.1 * dst.max()
            y, x = np.where(dst > thresh)
            coordinates = np.float32(np.stack((x, y), axis=-1))

            if self.debug:
                for i in range(len(x)):
                    cv2.circle(harris_image, (x[i], y[i]), 3, (0, 255, 0), -1)  # Red circles, filled
            
            # Skip clustering if no corners were detected
            if len(coordinates) == 0:
                print(f"Warning: No corners detected for contour {contour['id']}")
                contour['corners'] = np.array([])
                continue
                
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates)
            dbscan = DBSCAN(eps=0.07, min_samples=3)
            labels = dbscan.fit_predict(coordinates_scaled)
            unique_labels = set(labels)
            centers = []
            for label in unique_labels:
                if label != -1:  # Ignore noise points (-1)
                    cluster_points = coordinates[labels == label]
                    center = np.mean(cluster_points, axis=0)
                    centers.append(center)
            centers = np.array(centers)
            
            # If no valid clusters were found, use the original corner points
            if len(centers) == 0:
                print(f"Warning: No valid clusters found for contour {contour['id']}, using original corners")
                # Use a subset of original corners to avoid too many points
                step = max(1, len(coordinates) // 10)  # Take at most 10 corners
                centers = coordinates[::step]
            
            contour['corners'] = centers
            
        if self.debug:
            self.show_image(harris_image, 'Harris Corners')
        corners_image = contour_image.copy()   
        # Draw circles at cluster centers
        for contour in contours:
            for center in contour['corners']:
                cv2.circle(corners_image, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
        if self.debug:
            self.show_image(corners_image, 'Corners')

        for i, bbox in enumerate(resized_bboxes):
            if bbox['class'] in self.non_components:
                continue
            possible_contours = []
            rects_image = corners_image.copy()
            xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
            xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
    
            for contour in contours:
                c_xmin = contour['rectangle'][0]
                c_ymin = contour['rectangle'][1]
                c_xmax = c_xmin + contour['rectangle'][2]
                c_ymax = c_ymin + contour['rectangle'][3]
                cv2.rectangle(rects_image, (c_xmin, c_ymin), (c_xmax, c_ymax), color=(0, 255, 0), thickness=1)
    
                if ymax < c_ymin or ymin > c_ymax or xmax < c_xmin or xmin > c_xmax:
                    continue
                else:
                    possible_contours.append(contour)
                    for point in contour['corners']:
                        if self.is_point_near_bbox(point, bbox, pixel_threshold=6):
                            if original_size:
                                nodes[contour['id']]['components'].append(bboxes[i])
                            else:
                                nodes[contour['id']]['components'].append(bbox)
                            break
        if self.debug:
            self.show_image(rects_image, "Nodes")

        # Filter out empty nodes
        valid_nodes = {node_id: node_data for node_id, node_data in nodes.items() 
                      if node_data['components']}
        
        # Find node with most connections
        max_connections = max(len(node_data['components']) for node_data in valid_nodes.values())
        nodes_with_max = [node_id for node_id, node_data in valid_nodes.items() 
                         if len(node_data['components']) == max_connections]

        if len(nodes_with_max) == 1:
            # If there's only one node with max connections, it becomes node 0
            old_id = nodes_with_max[0]
        else:
            # If multiple nodes have same number of connections, find the one connected to source
            source_connected_nodes = []
            for node_id in nodes_with_max:
                components = valid_nodes[node_id]['components']
                if any(comp['class'] == 'source' for comp in components):
                    source_connected_nodes.append((node_id, 
                                                # Get y-coordinate of the node
                                                next(contour['rectangle'][1] 
                                                     for contour in contours 
                                                     if contour['id'] == node_id),
                                                # Get x-coordinate of the node
                                                next(contour['rectangle'][0] 
                                                     for contour in contours 
                                                     if contour['id'] == node_id)))
    
            if source_connected_nodes:
                # Sort by y-coordinate (descending) then x-coordinate
                source_connected_nodes.sort(key=lambda x: (-x[1], x[2]))
                old_id = source_connected_nodes[0][0]
            else:
                # If no source connections, use the first node
                old_id = nodes_with_max[0]

        # Rename the chosen node to 0 and adjust other node numbers
        new_nodes = []
        used_ids = set()
    
        # First add the node that should be 0
        new_nodes.append({
            'id': 0,
            'components': valid_nodes[old_id]['components'],
            'contour': valid_nodes[old_id]['contour']
        })
        used_ids.add(old_id)
    
        # Then add all other nodes
        current_id = 1
                
        for node_id, node_data in valid_nodes.items():
            if node_id not in used_ids and len(node_data['components']) >= 2:
                new_nodes.append({
                    'id': current_id,
                    'components': node_data['components'],
                    'contour': node_data['contour']
                })
                current_id += 1
        

        # Create visualization of final node numbering
        final_visualization = emptied_mask.copy()
        if len(final_visualization.shape) == 2:
            final_visualization = cv2.cvtColor(final_visualization, cv2.COLOR_GRAY2BGR)
        
        for node in new_nodes:
            # Find centroid of contour
            M = cv2.moments(node['contour'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                # Draw contour
                cv2.drawContours(final_visualization, [node['contour']], -1, (0, 255, 0), 2)

                # Draw node number
                cv2.putText(final_visualization, str(node['id']), 
                            (cx-10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 0, 255), 2)

        return new_nodes, emptied_mask, enhanced, contour_image, corners_image, final_visualization

    


    def generate_netlist_from_nodes(self, node_list):
        netlist = []
        id_count = 1
        component_counters = {component_type: 1 for component_type in set(self.netlist_map.values()) if component_type}
        processed_components = set()
        
        def get_component_id(component_type):
            component_id = f"{component_type}{component_counters[component_type]}"
            component_counters[component_type] += 1
            return component_id
            
        # First, find source components and their node connections
        source_nodes = {}  # Maps nodes to their relationship with sources (positive/negative)
        source_positive_node = None
        
        # First pass - find source and its positive node
        for n in node_list:
            node_components = n['components']
            node_index = n['id']
            for component in node_components:
                if component.get('class') in ['source', 'vdd']:
                    component_position = (component['xmin'], component['ymin'], component['xmax'], component['ymax'])
                    if component_position not in processed_components:
                        processed_components.add(component_position)
                        
                        # Find the other node this source is connected to
                        other_node = None
                        for other_index, other_components in enumerate(node_list):
                            if other_index != node_index:
                                if any(c['xmin'] == component['xmin'] and 
                                      c['ymin'] == component['ymin'] and 
                                      c['xmax'] == component['xmax'] and 
                                      c['ymax'] == component['ymax'] for c in other_components):
                                    other_node = other_index
                                    break
                        
                        if other_node is not None:
                            # The higher y-coordinate terminal is negative (connects to ground/node 0)
                            if component['ymax'] > component['ymin']:
                                source_positive_node = node_index
                            else:
                                source_positive_node = other_node
        
        # Reset processed components for main pass
        processed_components.clear()
        
        # Process components
        for n in node_list:
            node_components = n['components']
            node_index = n['id']
            node_id = node_index
            
            for component in node_components:
                component_class = component.get('class')
                component_position = (component['xmin'], component['ymin'], component['xmax'], component['ymax'])
                
                if component_class in ['text', 'explanatory', 'junction', 'crossover', 'terminal'] or component_position in processed_components:
                    continue
                    
                processed_components.add(component_position)
                component_type = self.netlist_map.get(component_class, 'UN')
                
                if not component_type:
                    continue
                    
                component_id = get_component_id(component_type)
                
                # Find the other node this component is connected to
                other_node = None
                for n2 in node_list:
                    other_components = n2['components']
                    other_index = n2['id']
                    
                    if other_index != node_index:
                        if any(c['xmin'] == component['xmin'] and 
                              c['ymin'] == component['ymin'] and 
                              c['xmax'] == component['xmax'] and 
                              c['ymax'] == component['ymax'] for c in other_components):
                            other_node = other_index
                            break
                            
                if other_node is None:
                    if self.debug:
                        print(f"Warning: Could not find second node for component {component_id}")
                    continue
                
                value = "None"  # Placeholder value
                
                component['id'] = id_count
                id_count += 1
                
                node_1, node_2 = node_id, other_node
                
                # For source components
                if component_class in ['source', 'vdd']:
                    # Always put positive node first, node 0 second
                    if source_positive_node == node_id:
                        node_1, node_2 = node_id, 0
                    else:
                        node_1, node_2 = other_node, 0
                elif component_class in ['gnd', 'vss']:
                    # Ground always connects to 0 second
                    node_1, node_2 = node_id, 0
                else:
                    # For other components
                    # If either node is 0, make sure it's second
                    if node_1 == 0:
                        node_1, node_2 = node_2, node_1
                    elif node_2 == 0:
                        pass  # node 0 is already second
                    # Otherwise, try to maintain current flow direction
                    elif node_1 == source_positive_node or node_2 == source_positive_node:
                        if node_2 == source_positive_node:
                            node_1, node_2 = node_2, node_1
                
                line = {'component_type': component_type,
                       'component_num': component_counters[component_type]-1,
                       'node_1': node_1,
                       'node_2': node_2,
                       'value': value}
                line.update(component)
                netlist.append(line)
        
        return netlist

    def fix_netlist(self, netlist, vlm_out):
        for line in netlist:
            for item in vlm_out:
                if str(item['id']) == str(line['id']):
                    if line['value'] == None or line['value'] == 'None':
                        line['value'] = item['value']

                    if line['class'] == 'unknown':
                        line['component_num'] = [int(l.get("component_num", 0)) for l in netlist if l['class'] == item['class']]
                        line['component_num'].append(0)
                        line['component_num'] = max(line['component_num']) + 1
                        line['class'] = item['class']
                        line['component_type'] = self.netlist_map[line['class']]
                    elif line['class'] != item['class']:
                        if self.debug:
                            print(f"Mismatch between VLM output and YOLO output for component {line['id']}, VLM prioritized.\n{line['class']} -> {item['class']}")
                        line['class'] = 'unknown'
                        line['component_num'] = [int(l.get("component_num", 0)) for l in netlist if l['class'] == item['class']]
                        line['component_num'].append(0)
                        line['component_num'] = max(line['component_num']) + 1
                        line['class'] = item['class']
                        line['component_type'] = self.netlist_map[line['class']]
                    break
                else:
                    continue
    
    
    def stringify_line(self, netlist_line):
        return f"{netlist_line['component_type']}{netlist_line['component_num']} {netlist_line['node_1']} {netlist_line['node_2']} {netlist_line['value']}"
```

```src\utills.py
# System Imports
import os
from dotenv import load_dotenv
import sys, shutil
from os.path import join, realpath
import json
import xml.etree.ElementTree as ET
from xml import etree
from random import choice
import random
from copy import deepcopy
import cv2
from google import genai
from google.genai import types
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom
import ast
import re
from typing import Union, Dict
import streamlit as st


# Load environment variables from .env file
load_dotenv()

components_dict = {
    'gnd': 'Ground: A reference point in an electrical circuit. has no direction nor value (both None).',
    'voltage.ac': 'AC Voltage source, has no direction (None). if its value is written in phasor, write it as magnitude:phase',
    'voltage.dc': 'DC Voltage source, has a direction, should be specified as either up, down, left or right, depending on the direction of the + sign',
    'voltage.battery': 'Battery Voltage source, has the direction of the largest horizontal bar of its symbol (up, down, left or right)',
    'resistor': 'Resistor: A passive component with no direction (None)',
    'voltage.dependent': 'Voltage-Dependent Source: A voltage source whose output voltage depends on another voltage or current in the circuit. has the direction of the + sign (up, down, left or right).',
    'current.dc': 'DC Current: Direct current, where the current flows in one direction consistently, has the direction of the arrow inside the symbol (up, down, left or right).',
    'current.dependent': 'Current-Dependent Source: A current source whose output current depends on another current or voltage in the circuit. has the direction of the arrow inside the symbol (up, down, left or right).',
    'capacitor': 'Capacitor: A passive component with no direction (None)',
    'inductor': 'Inductor: A passive component with no direction (None)',
    'diode': 'Diode: has the direction of the symbol which is like an arrow (up, down, left or right)'
}



def load_classes() -> dict:
    """Returns the List of Classes as Encoding Map"""

    with open("classes.json") as classes_file:
        return json.loads(classes_file.read())


def summarize_components(component_list):
    """Generates a concise summary of detected electrical components."""
    summary = {}
    for c in component_list:
        class_name = c['class'].replace('.', ' ').title()
        summary[class_name] = summary.get(class_name, 0) + 1 # Count occurrences efficiently


    summary_string = "Detected: "
    for class_name, count in summary.items():
        summary_string += f"{count} {class_name}{'s' if count > 1 else ''}, "  # Add 's' for plurals


    return summary_string[:-2]  # Remove trailing comma and space


def gemini_labels(image_file):
    """
    Uploads an image of a circuit schematic to the Gemini API, retrieves 
    the components and their values, and returns them in a structured format.

    Args:
        image_file (np.ndarray): Image array of the circuit schematic.

    Returns:
        dict: A dictionary with component names as keys and a list of their
              values as entries.
    """
    # Load API key from Streamlit secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        raise ValueError("GEMINI_API_KEY not found in Streamlit secrets")
        
    client = genai.Client(api_key=api_key)
    
    # Convert the image file to PIL Image format if it's a numpy array
    image_file = Image.fromarray(image_file) 
    
    # MODEL = "gemini-2.5-flash-preview-04-17"
    MODEL = "gemini-2.5-pro-exp-03-25"
    
    prompt = ("Identify only the components and their values in this circuit schematic, the id of each component is in red. return the object as a python list of dictioaries."
              "If the values are just letters or don't exist for the component, the value should be None. If components are defined using complex values, write the complex/imaginary value."
              "Format the output as a list of dictionaries [{'class': 'voltage.dependent', 'value':'35*V_2', 'direction':'down', 'id': '1'}, "
              "{'class': 'voltage.ac', 'value':'22k:30', 'direction': 'None', 'id': '2'}], note how you always use the same ids of the components marked in red in the image, to identify the component. the closest red number to a component that component's id. nothing else. The value of the component should not specify the unit, only the suffix such as k or M or m or nothing, there shouldn't be a space between the number and the suffix. The classes included and their descriptions: " 
              + str(components_dict))
    
    # Generate content using the image and prompt
    response = client.models.generate_content(
        model=MODEL,
        contents=[image_file, "\n\n", prompt],
        config=types.GenerateContentConfig(
            temperature=0.0
            )
    )
    print(response.text)
    
    formatted = response.text.strip('```python\n')
    formatted = formatted.strip('```json\n')
    formatted = formatted.strip('```')
    parsed_data = ast.literal_eval(formatted)
#     for line in parsed_data:
#         line['value'] = parse_value(line['value'])
    return parsed_data

def show_image(img, title="Image"):
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def calculate_iou(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bbox1 (tuple): Bounding box 1 (xmin, ymin, xmax, ymax).
        bbox2 (tuple): Bounding box 2 (xmin, ymin, xmax, ymax).

    Returns:
        float: IoU value (0.0 <= IoU <= 1.0).
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1['xmin'], bbox1['ymin'], bbox1['xmax'], bbox1['ymax']
    xmin2, ymin2, xmax2, ymax2 = bbox2['xmin'], bbox2['ymin'], bbox2['xmax'], bbox2['ymax']

    
    # Calculate intersection area
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    inter_width = max(inter_xmax - inter_xmin, 0)
    inter_height = max(inter_ymax - inter_ymin, 0)
    intersection_area = inter_width * inter_height

    # Calculate union area
    bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def non_max_suppression_by_area(bboxes, iou_threshold=0.5):
    """Applies NMS based on area for bounding boxes in dictionary format."""

    # Sort by area (accessing 'xmin', 'xmax', etc. from dictionaries)
    bboxes = sorted(bboxes, key=lambda bbox: (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin']), reverse=True)

    filtered_bboxes = []
    while bboxes:
        bbox1 = bboxes.pop(0)
        filtered_bboxes.append(bbox1)

        bboxes = [bbox2 for bbox2 in bboxes
                  if calculate_iou(bbox1, bbox2) < iou_threshold]

    return filtered_bboxes

def non_max_suppression_by_confidence(bboxes, iou_threshold=0.5):
    """Applies NMS based on confidence for bounding boxes in dictionary format."""

    # Sort by confidence score (descending)
    bboxes = sorted(bboxes, key=lambda bbox: bbox['confidence'], reverse=True)

    filtered_bboxes = []
    while bboxes:
        bbox1 = bboxes.pop(0)
        filtered_bboxes.append(bbox1)

        # Keep boxes that have low IoU with the current highest confidence box
        bboxes = [bbox2 for bbox2 in bboxes
                  if calculate_iou(bbox1, bbox2) < iou_threshold]

    return filtered_bboxes

def parse_component_value(value: str) -> Union[float, complex]:
    """
    Extremely robust component value parser with advanced scientific notation support.
    
    Handles multiple scientific notation formats:
    - 5x10^-5
    - 5e-5
    - 5E-5
    - 5 x 10^-5
    - 5 * 10^-5
    
    Supports:
    - Numeric formats: integers, floats, scientific notation
    - Unit suffixes for various components
    - Complex numbers
    - Multiple delimiter styles
    
    Args:
        value (str): Component value string
    
    Returns:
        Union[float, complex]: Parsed numeric value
    """
    # Normalize input
    value_str = str(value).strip().lower().replace(' ', '')
    
    # Comprehensive prefix mapping (metric prefixes)
    prefix_map = {
        'y': 1e-24,   # yocto
        'z': 1e-21,   # zepto
        'a': 1e-18,   # atto
        'f': 1e-15,   # femto
        'p': 1e-12,   # pico
        'n': 1e-9,    # nano
        'u': 1e-6,    # micro
        'm': 1e-3,    # milli
        'c': 1e-2,    # centi
        'd': 1e-1,    # deci
        'k': 1e3,     # kilo
        'M': 1e6,     # mega
        'G': 1e9,     # giga
        'T': 1e12,    # tera
        'P': 1e15,    # peta
        'E': 1e18,    # exa
        'Z': 1e21,    # zetta
        'Y': 1e24     # yotta
    }
    
    # Comprehensive unit mapping
    unit_map = {
        'Ω': None, 'ohm': None, 'r': None,  # Resistors
        'f': 'c', 'farad': 'c', 'c': 'c',   # Capacitors
        'h': 'l', 'henry': 'l', 'l': 'l',   # Inductors
        'v': 'v', 'volt': 'v',              # Voltage
        'a': 'a', 'ampere': 'a'             # Current
    }
    
    # Scientific notation patterns
    sci_notation_patterns = [
        # 5x10^-5 | 5 x 10^-5 | 5 * 10^-5
        r'^(\d+\.?\d*)\s*[x*]\s*10\^(-?\d+)$',
        
        # 5e-5 | 5E-5
        r'^(\d+\.?\d*)[eE](-?\d+)$'
    ]
    
    # Complex number detection
    complex_patterns = [
        r'^([-+]?\d*\.?\d+)\s*([+-]?\s*j\d*\.?\d*)$',  # 5+j3, 5-j3
        r'^([-+]?\d*\.?\d+)\s*([+-]?\s*\d*\.?\d*)j$',  # 5+3j, 5-3j
        r'^([-+]?j\d*\.?\d*)$',                        # j5, -j3
    ]
    
    # Try complex number parsing first
    for pattern in complex_patterns:
        match = re.match(pattern, value_str)
        if match:
            try:
                return complex(match.group(1).replace(' ', '') + match.group(2).replace(' ', ''))
            except ValueError:
                pass
    
    # Try scientific notation parsing
    for pattern in sci_notation_patterns:
        match = re.match(pattern, value_str)
        if match:
            try:
                base = float(match.group(1))
                exponent = int(match.group(2))
                return base * (10 ** exponent)
            except ValueError:
                pass
    
    # Detect and remove unit
    unit_match = None
    multiplier = 1
    
    # Remove known units
    for unit_pattern, unit_type in unit_map.items():
        if value_str.endswith(unit_pattern):
            unit_match = unit_type
            value_str = value_str[:-len(unit_pattern)]
            break
    
    # Detect and apply prefix multiplier
    for prefix, mult in prefix_map.items():
        if value_str.startswith(prefix):
            multiplier = mult
            value_str = value_str[len(prefix):]
            break
    
    # Final parsing attempt
    try:
        # Handle any remaining scientific notation or simple float
        parsed_value = float(value_str) * multiplier
        return parsed_value
    except ValueError:
        raise ValueError(f"Could not parse value: {value}")
```

```src\views.py
import numpy as np
import cv2
import os
import shutil
from .utills import summarize_components, gemini_labels
from .circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.Parser import SpiceParser
from PySpice.Spice.Simulation import CircuitSimulation
from pathlib import Path

analyzer = CircuitAnalyzer(yolo_path='models/YOLO/best_large_model_yolo.pt', debug=False)
files_location = 'static/assets/uploads/'
components_dict = {
    'gnd': 'Ground: A reference point in an electrical circuit. has no direction nor value (both None).',
    'voltage.ac': 'AC Voltage source, has no direction (None). if its value is written in phasor, write it as magnitude:phase',
    'voltage.dc': 'DC Voltage source, has a direction, should be specified as either up, down, left or right, depending on the direction of the + sign',
    'voltage.battery': 'Battery Voltage source, has the direction of the largest horizontal bar of its symbol (up, down, left or right)',
    'resistor': 'Resistor: A passive component with no direction (None)',
    'voltage.dependent': 'Voltage-Dependent Source: A voltage source whose output voltage depends on another voltage or current in the circuit. has the direction of the + sign (up, down, left or right).',
    'current.dc': 'DC Current: Direct current, where the current flows in one direction consistently, has the direction of the arrow inside the symbol (up, down, left or right).',
    'current.dependent': 'Current-Dependent Source: A current source whose output current depends on another current or voltage in the circuit. has the direction of the arrow inside the symbol (up, down, left or right).',
    'capacitor': 'Capacitor: A passive component with no direction (None)',
    'inductor': 'Inductor: A passive component with no direction (None)',
    'diode': 'Diode: has the direction of the symbol which is like an arrow (up, down, left or right)'
}


```

```src\sam2_infer.py
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    #device = torch.device("xla")
print(f"using device: {device}")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = r"D:\SDP_demo\models\SAM2\sam2.1_hiera_large.pt"
model_cfg = r"D:\SDP_demo\models\configs\sam2.1_hiera_l.yaml"

sam2_model = build_sam2(
    model_cfg,
    sam2_checkpoint,
    device=device,
    mode="train"
)
sam2_model.use_high_res_features = True

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks
    
_transforms = SAM2Transforms(
            resolution=sam2_model.image_size,
            mask_threshold=0,
            max_hole_area=0,
            max_sprinkle_area=0,
        )

from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from peft import LoraConfig, get_peft_model, TaskType # Import necessary peft components

class MultiKernelRefinement(nn.Module):
    """
    Applies multiple convolutional kernels in parallel for refinement
    and combines their outputs.
    """
    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=[3, 5, 7, 9, 11], intermediate_channels=8):
        """
        Args:
            in_channels (int): Number of input channels (usually 1 for mask logits).
            out_channels (int): Number of final output channels (usually 1 for refined logits).
            kernel_sizes (list[int]): List of odd kernel sizes for parallel conv branches.
            intermediate_channels (int): Number of output channels for EACH parallel conv branch.
        """
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.intermediate_channels_per_branch = intermediate_channels

        # Create parallel convolutional branches
        self.conv_branches = nn.ModuleList()
        for k_size in kernel_sizes:
            # padding='same' ensures output H, W match input H, W for odd kernels
            # For even kernels, padding needs manual calculation: padding = (k_size - 1) // 2
            if k_size % 2 == 0:
                 raise ValueError(f"Even kernel size {k_size} not directly supported with padding='same'. Use odd kernels or calculate padding manually.")
            branch = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.intermediate_channels_per_branch,
                kernel_size=k_size,
                padding='same', # Works for odd kernel sizes
                bias=True
            )
            self.conv_branches.append(branch)

        # Activation function after each branch (optional but common)
        self.activation = nn.GELU() # Or nn.GELU() etc.

        # Final combination layer
        # Takes concatenated features from all branches
        total_intermediate_channels = len(kernel_sizes) * self.intermediate_channels_per_branch
        self.combiner_conv = nn.Conv2d(
            in_channels=total_intermediate_channels,
            out_channels=out_channels,
            kernel_size=1, # 1x1 convolution to combine features channel-wise
            padding=0,
            bias=True
        )

    def forward(self, x):
        branch_outputs = []
        for branch_conv in self.conv_branches:
            branch_out = self.activation(branch_conv(x)) # Apply conv then activation
            branch_outputs.append(branch_out)

        # Concatenate outputs along the channel dimension
        concatenated_features = torch.cat(branch_outputs, dim=1) # Shape: (B, total_intermediate_channels, H, W)

        # Combine features using the 1x1 convolution
        refined_output = self.combiner_conv(concatenated_features) # Shape: (B, out_channels, H, W)

        return refined_output

class SAM2ImageWrapper(nn.Module):
    """
    A wrapper around SAM2Base for image-only segmentation.
    Applies LoRA internally to the wrapped model.
    """
    def __init__(self, modified_sam2_model: SAM2Base, embedding_r=4, use_refinement=False, refinement_kernel_sizes=[3, 5, 7, 9, 11]):
        super().__init__()
        self.sam2_model = modified_sam2_model
        self.use_refinement = use_refinement
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        
        self.embedding_r = embedding_r
        self.dense_embedding1 = nn.Parameter(torch.randn(1, 256, self.embedding_r))
        self.dense_embedding2 = nn.Parameter(torch.randn(1, self.embedding_r, 64 * 64))
        self.sparse_embedding = nn.Parameter(torch.randn(1, 32, 256))
        if self.use_refinement:
            self.refinement_layer = MultiKernelRefinement(
                in_channels=1,
                out_channels=1,
                kernel_sizes=refinement_kernel_sizes, # Example kernel sizes
                intermediate_channels=4 # Example intermediate channels per branch
            )
        else:
            self.refinement_layer = None

    def forward(self, images, points=None, point_labels=None, masks_prompt=None, multimask_output=False):
        """
        Simplified forward pass using the wrapped SAM2Base's methods.
        """

        # 1. Encode Image
        out = self.sam2_model.image_encoder(images)
        out["backbone_fpn"][0] = self.sam2_model.sam_mask_decoder.conv_s0(
            out["backbone_fpn"][0]
        )
        out["backbone_fpn"][1] = self.sam2_model.sam_mask_decoder.conv_s1(
            out["backbone_fpn"][1]
        )

        # 2. Prepare Decoder Inputs
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(out)
        # --- Corrected List Comprehension ---
        feats = [
            # Get Batch Size (B) dynamically from the input feature tensor
            feat.permute(1, 2, 0).view(feat.shape[1], -1, *feat_size)
            #                            ^^^^^^^^^^^  Use B from feat.shape[1]
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1] # Reverse the resulting list
        # --- Dictionary Creation (remains the same) ---
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        # 4. Run SAM Prompt Encoder and Mask Decoder directly
        high_res_features = _features["high_res_feats"]

        # compute the trainable prompt embedding
        dense_embedding = (self.dense_embedding1 @ self.dense_embedding2).view(1, 256, 64, 64)
        
        low_res_masks, iou_predictions, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=_features["image_embed"],
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=self.sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            multimask_output=False,
            repeat_image=True,
            high_res_features=high_res_features,
        )

        # 5. Return desired outputs
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(self.sam2_model.image_size, self.sam2_model.image_size),
            mode="bilinear",
            align_corners=False,
        )


        if self.use_refinement:
            high_res_masks = self.refinement_layer(high_res_masks)
            
        
        return high_res_masks, low_res_masks, iou_predictions
    
def get_modified_sam2(
    # --- Model Config ---
    model_cfg_path: str = r"D:\SDP_demo\models\configs\sam2.1_hiera_l.yaml",
    checkpoint_path: str = r"D:\SDP_demo\models\SAM2\sam2.1_hiera_large.pt",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    use_high_res_features: bool = True,
    # --- PEFT Config ---
    use_peft: bool = True,
    lora_rank: int = 12,
    lora_alpha: int = 16,
    lora_dropout: float = 0.2,
    lora_target_modules: list = None, # List of module names (strings)
    # --- Wrapper/Task Config ---
    use_wrapper: bool = True,
    trainable_embedding_r: int = 4,
    # --- Refinement Layer ---
    use_refinement_layer: bool = False,
    refinement_kernels: list = [3, 5, 7, 11],
    kernel_channels: int = 4,
    # --- Loss Settings ---
    weight_dice=0.5, weight_focal=0.4, weight_iou=0.1, 
    weight_tversky: float = 0.0, weight_tv: float = 0.0, weight_freq: float = 0.0,
    dice_smooth=1e-5, focal_alpha=0.25, focal_gamma=2.0,
    iou_smooth=1e-5, iou_threshold=0.5,
    tversky_alpha = 0.2, tversky_beta= 0.8,
    apply_sigmoid=True,
    # --- Optimizer Settings ---
    lr=1e-3
    ):
    """
    Initializes SAM 2, applies PEFT/LoRA, and optionally wraps it for image-only tasks.

    Args:
        model_cfg_path (str): Path to the SAM 2 model config YAML.
        checkpoint_path (str): Path to the SAM 2 model checkpoint (.pt file).
        device (str): Device to load the model onto ('cuda', 'cpu', 'mps').
        use_high_res_features (bool): Whether the decoder should use high-res skip connections.
        use_peft (bool): Whether to apply PEFT/LoRA.
        lora_rank (int): Rank for LoRA matrices.
        lora_alpha (int): Alpha scaling for LoRA.
        lora_dropout (float): Dropout probability for LoRA layers.
        lora_target_modules (list): List of specific module names (strings) within the original
                                     SAM2 model to apply LoRA to. If None, PEFT might guess or
                                     you might need to define defaults.
        use_wrapper (bool): Whether to wrap the (PEFT-)modified model in SAM2ImageWrapper.
        trainable_embedding_r (int): Rank factor for the trainable prompt embeddings in the wrapper.
        use_refinement_layer (bool): Whether to add the MultiKernelRefinement layer in the wrapper.

    Returns:
        torch.nn.Module: The potentially PEFT-modified and wrapped SAM 2 model.
    """
    print("--- Initializing Modified SAM 2 ---")
    model_device = torch.device(device)

    # 1. Load Original SAM 2 Model
    print(f"Loading SAM 2 from config: {model_cfg_path} and checkpoint: {checkpoint_path}")
    original_sam2_model = build_sam2(
        model_cfg_path,
        checkpoint_path,
        device=model_device,
        mode="train" # Keep in train mode for fine-tuning
    )
    original_sam2_model.use_high_res_features_in_sam = use_high_res_features
    print(f"Original model loaded on {model_device}. use_high_res_features_in_sam set to {use_high_res_features}.")

    # --- Model to be returned ---
    final_model = original_sam2_model

    # 2. Apply PEFT/LoRA if enabled
    if use_peft:
        print(f"Applying PEFT/LoRA with rank={lora_rank}, alpha={lora_alpha}")
        if lora_target_modules is None:
            # Define default target modules if none provided
            # These defaults should cover key areas for image-only fine-tuning
            lora_target_modules = [
                # Decoder Transformer Attention (Self and Cross)
                "sam_mask_decoder.transformer.layers.0.self_attn.k_proj",
            ]
            print(f"Using default lora_target_modules: {lora_target_modules}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none", # Common setting
            modules_to_save=None, # Only train LoRA parameters
            init_lora_weights=True, # Default initialization
        )

        # Apply PEFT
        # Note: get_peft_model freezes non-target layers automatically
        peft_model = get_peft_model(original_sam2_model, lora_config)
        print("PEFT LoRA configuration applied.")
        peft_model.print_trainable_parameters()
        final_model = peft_model # Update the model to be returned
    else:
        print("Skipping PEFT/LoRA application. Freezing all parameters.")
        # Freeze all parameters if not using PEFT, as wrapper adds new ones
        for param in final_model.parameters():
             param.requires_grad = False
            
    # 3. Apply Wrapper if enabled
    if use_wrapper:
        print("Applying SAM2ImageWrapper...")
        wrapped_model = SAM2ImageWrapper(
            modified_sam2_model=final_model, # Pass the (potentially PEFT-modified) model
            embedding_r=trainable_embedding_r,
            use_refinement=use_refinement_layer,
            refinement_kernel_sizes=refinement_kernels
        )
        final_model = wrapped_model.to(model_device) # Update the model to be returned
        print("Wrapper applied.")
    else:
        print("Skipping SAM2ImageWrapper.")
        # If not using the wrapper, ensure the training loop correctly calls
        # the PEFT model with the image-only logic and handles prompts.

    # 4. Final Verification of Trainable Parameters
    print("\n--- Final Trainable Parameters ---")
    total_trainable = 0
    for name, param in final_model.named_parameters():
        if param.requires_grad:
            print(f"- {name}: {param.shape} ({param.numel()})")
            total_trainable += param.numel()
    print(f"Total Trainable Parameters in Final Model: {total_trainable}")
    if total_trainable == 0 and (use_peft or use_wrapper):
         print("WARNING: No trainable parameters found! Check PEFT config and wrapper parameter initialization.")
    elif not use_peft and not use_wrapper and total_trainable > 0:
         print("Warning: Model has trainable parameters but PEFT/Wrapper were not used?")


    print("--- Optimizer Is Ready ---")
    return final_model

base_parts = ["sam_mask_decoder.transformer.layers.0.self_attn.k_proj",
                "sam_mask_decoder.transformer.layers.0.self_attn.q_proj",
                "sam_mask_decoder.transformer.layers.0.self_attn.v_proj",
                "sam_mask_decoder.transformer.layers.0.self_attn.out_proj",
                "sam_mask_decoder.transformer.layers.1.self_attn.k_proj",
                "sam_mask_decoder.transformer.layers.1.self_attn.q_proj",
                "sam_mask_decoder.transformer.layers.1.self_attn.v_proj",
                "sam_mask_decoder.transformer.layers.1.self_attn.out_proj",
                "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.k_proj",
                "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj",
                "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj",
                "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.out_proj",
                "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.k_proj",
                "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj",
                "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj",
                "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.out_proj",
                "sam_mask_decoder.transformer.layers.0.mlp.layers.0",
                "sam_mask_decoder.transformer.layers.0.mlp.layers.1",
                "sam_mask_decoder.transformer.layers.1.mlp.layers.0",
                "sam_mask_decoder.transformer.layers.1.mlp.layers.1"]
added_parts = [
            "sam_mask_decoder.iou_prediction_head.layers.2",
             "sam_mask_decoder.conv_s0",
                "sam_mask_decoder.conv_s1",
             
             "image_encoder.neck.convs.2.conv",
            "image_encoder.neck.convs.3.conv", 
             
             "image_encoder.trunk.blocks.44.attn.qkv", # for downsampling
              "image_encoder.trunk.blocks.44.mlp.layers.0",
              "image_encoder.trunk.blocks.44.proj",

            "image_encoder.trunk.blocks.47.attn.qkv",
            "image_encoder.trunk.blocks.47.mlp.layers.0",
            
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.k_proj",
            "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj",
            "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj",
            "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.k_proj",
            "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj",]

modified_sam2 = get_modified_sam2(
    lora_rank = 4,
    lora_alpha = 16,
    weight_dice= 0.5, weight_focal=0.4, weight_iou=0.3, weight_freq=0.1,
    use_refinement_layer = True,
    refinement_kernels = [3, 5, 7, 11],
    kernel_channels = 2,
    focal_alpha=0.25,
    lora_target_modules= base_parts + added_parts,
    lora_dropout = 0.3,
    lr=1e-3
)

ckp_path = r'D:\SDP_demo\models\SAM2\best_miou_model_SAM_latest.pth'
checkpoint = torch.load(ckp_path, map_location=device, weights_only=True) 
model_state_dict = checkpoint['state_dict'] 
modified_sam2.load_state_dict(model_state_dict)
```

```app.py
import torch
import numpy as np
import cv2
import shutil
import streamlit as st
from src.utills import (summarize_components,
                        gemini_labels,
                        non_max_suppression_by_confidence)
from src.circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Spice.Parser import SpiceParser
from pathlib import Path

torch.classes.__path__ = []

# Set page config and custom styles
st.set_page_config(
    page_title="Circuit Analyzer",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Set up base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'static/uploads'
YOLO_MODEL_PATH = BASE_DIR / 'models/YOLO/best_large_model_yolo.pt'
# Add SAM2 paths
SAM2_CONFIG_PATH = Path(r"D:\SDP_demo\models\configs\sam2.1_hiera_l.yaml")
SAM2_BASE_CHECKPOINT_PATH = Path(r"D:\SDP_demo\models\SAM2\sam2.1_hiera_large.pt")  # Base model
SAM2_FINETUNED_CHECKPOINT_PATH = Path(r"D:\SDP_demo\models\SAM2\best_miou_model_SAM_latest.pth")  # Fine-tuned weights

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'models/SAM2').mkdir(parents=True, exist_ok=True)  # Ensure models/SAM2 exists

# Check if SAM2 model files exist
if not SAM2_CONFIG_PATH.exists() or not SAM2_BASE_CHECKPOINT_PATH.exists() or not SAM2_FINETUNED_CHECKPOINT_PATH.exists():
    missing_files = []
    if not SAM2_CONFIG_PATH.exists():
        missing_files.append(f"Config: {SAM2_CONFIG_PATH}")
    if not SAM2_BASE_CHECKPOINT_PATH.exists():
        missing_files.append(f"Base Checkpoint: {SAM2_BASE_CHECKPOINT_PATH}")
    if not SAM2_FINETUNED_CHECKPOINT_PATH.exists():
        missing_files.append(f"Fine-tuned Checkpoint: {SAM2_FINETUNED_CHECKPOINT_PATH}")
    
    st.warning(f"One or more SAM2 model files not found:\n" + "\n".join(missing_files) + "\nSAM2 features will be disabled.")
    use_sam2_feature = False
else:
    use_sam2_feature = True

# Initialize the circuit analyzer with error handling
@st.cache_resource
def load_circuit_analyzer():
    try:
        analyzer = CircuitAnalyzer(
            yolo_path=str(YOLO_MODEL_PATH),
            sam2_config_path=str(SAM2_CONFIG_PATH),
            sam2_base_checkpoint_path=str(SAM2_BASE_CHECKPOINT_PATH),
            sam2_finetuned_checkpoint_path=str(SAM2_FINETUNED_CHECKPOINT_PATH),
            use_sam2=use_sam2_feature,
            debug=False
        )
        return analyzer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Initialize the circuit analyzer
analyzer = load_circuit_analyzer()

# Check if analyzer loaded successfully before proceeding
if analyzer is None:
    st.stop()  # Stop the app if model loading failed

# Create containers for results
if 'results' not in st.session_state:
    st.session_state.results = {
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': None,
        'annotated_image': None,
        'component_stats': None,
        'node_visualization': None,
        'node_mask': None,
        'enhanced_mask': None,
        'contour_image': None,
        'corners_image': None
    }

# Track previous upload to prevent unnecessary resets
if 'previous_upload_name' not in st.session_state:
    st.session_state.previous_upload_name = None

# Main content
st.title("Circuit Diagram Analysis Tool")
st.markdown("Upload your circuit diagram and analyze it step by step.")

# Show SAM2 status
if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2:
    st.info("✅ SAM2 Segmentation is Enabled - Using advanced neural segmentation.")
else:
    st.warning("⚠️ SAM2 Segmentation is Disabled - Using traditional segmentation.")

# File upload section
uploaded_file = st.file_uploader(
    "Drag and drop your circuit diagram here",
    type=['png', 'jpg', 'jpeg'],
    help="For best results, use a clear image with good contrast"
)

if uploaded_file is not None:
    # Only reset results if a new file is uploaded
    if st.session_state.previous_upload_name != uploaded_file.name:
        # Convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        
        # Clear results when new image is uploaded
        st.session_state.results = {
            'bboxes': None,
            'nodes': None,
            'netlist': None,
            'netlist_text': None,
            'original_image': image,
            'uploaded_file_type': uploaded_file.type,
            'uploaded_file_name': uploaded_file.name,
            'annotated_image': None,
            'component_stats': None,
            'node_visualization': None,
            'node_mask': None,
            'enhanced_mask': None,
            'contour_image': None,
            'corners_image': None
        }
        
        # Store original image
        st.session_state.results['original_image'] = image
        
        # Clear and save files
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / f'1.{uploaded_file.name.split(".")[-1]}'
        save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(str(image_path), save_image)
        
        # Update previous upload name
        st.session_state.previous_upload_name = uploaded_file.name
    else:
        # If it's the same file, just ensure we have the image loaded
        if st.session_state.results['original_image'] is None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.results['original_image'] = image
    
    # Step 1: Image Analysis
    st.markdown("## Step 1: 📸 Image Analysis")
    analyze_container = st.container()

    # Always show the original image if it exists
    if st.session_state.results['original_image'] is not None:
        with analyze_container:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(st.session_state.results['original_image'], caption="Original Image", use_container_width=True)
            with col2:
                st.markdown("### Image Details")
                h, w = st.session_state.results['original_image'].shape[:2]
                st.markdown(f"- **Size**: {w}x{h} pixels")
                if 'uploaded_file_type' in st.session_state.results:
                    st.markdown(f"- **Format**: {st.session_state.results['uploaded_file_type']}")
                if 'uploaded_file_name' in st.session_state.results:
                    st.markdown(f"- **Name**: {st.session_state.results['uploaded_file_name']}")

    # Step 2: Component Detection
    st.markdown("## Step 2: 🔍 Component Detection")
    detection_container = st.container()
    
    # Always show previous detection results if they exist
    if st.session_state.results['annotated_image'] is not None:
        with detection_container:
            st.image(st.session_state.results['annotated_image'], caption="Component Detection with Confidence Scores", use_container_width=True)
            
            # Display component statistics
            if st.session_state.results['component_stats'] is not None:
                cols = st.columns(len(st.session_state.results['component_stats']))
                for col, (component, stats) in zip(cols, st.session_state.results['component_stats'].items()):
                    with col:
                        avg_conf = stats['total_conf'] / stats['count']
                        st.metric(
                            label=component,
                            value=stats['count'],
                            delta=f"Avg Conf: {avg_conf:.2f}"
                        )
    
    if st.button("Detect Components"):
        print("Detect Components button clicked")
        if st.session_state.results['original_image'] is None:
            print("No original image found in session state")
            st.warning("Please upload an image first")
        else:
            print("Original image found in session state")
            with st.spinner("Detecting components..."):
                # Get raw bounding boxes
                raw_bboxes = analyzer.bboxes(st.session_state.results['original_image'])
                print(f"Found {len(raw_bboxes)} raw bounding boxes")
                
                # Apply Non-Maximum Suppression
                bboxes = non_max_suppression_by_confidence(raw_bboxes, iou_threshold=0.6)
                print(f"After NMS: {len(bboxes)} bounding boxes")
                st.session_state.results['bboxes'] = bboxes # Store filtered bboxes
                print("Stored bboxes in session state")
                
                # Summarize based on filtered bboxes
                detection_summary = summarize_components(bboxes)
                
                with detection_container:
                    # Display annotated image with confidence scores using filtered bboxes
                    annotated_image = st.session_state.results['original_image'].copy()
                    for bbox in bboxes:
                        xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
                        xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])
                        label = bbox['class']
                        conf = bbox['confidence']
                        
                        # Draw rectangle
                        cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        
                        # Add label with confidence score
                        label_text = f"{label}: {conf:.2f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                        
                        # Draw white background for text
                        cv2.rectangle(annotated_image, 
                                    (xmin, ymin - text_height - 5),
                                    (xmin + text_width, ymin),
                                    (255, 255, 255),
                                    -1)
                        
                        # Draw text
                        cv2.putText(annotated_image, 
                                  label_text,
                                  (xmin, ymin - 5), 
                                  font,
                                  font_scale,
                                  (0, 0, 255),
                                  thickness)
                    
                    st.session_state.results['annotated_image'] = annotated_image
                    st.image(annotated_image, caption="Component Detection with Confidence Scores", use_container_width=True)
                    
                    # Parse and display component counts
                    component_stats = {}
                    for bbox in bboxes:
                        name = bbox['class']
                        conf = bbox['confidence']
                        if name not in component_stats:
                            component_stats[name] = {'count': 0, 'total_conf': 0}
                        component_stats[name]['count'] += 1
                        component_stats[name]['total_conf'] += conf
                    
                    st.session_state.results['component_stats'] = component_stats
                    
                    cols = st.columns(len(component_stats))
                    for col, (component, stats) in zip(cols, component_stats.items()):
                        with col:
                            avg_conf = stats['total_conf'] / stats['count']
                            st.metric(
                                label=component,
                                value=stats['count'],
                                delta=f"Avg Conf: {avg_conf:.2f}"
                            )

    # Step 3: Node Analysis
    st.markdown("## Step 3: 🔗 Node Analysis")
    node_container = st.container()
    
    # Debug prints
    print("Current state of results:", {k: "Present" if v is not None else "None" for k, v in st.session_state.results.items()})
    
    # Always show previous node analysis results if they exist
    if st.session_state.results['node_visualization'] is not None:
        print("Showing previous node analysis results")
        with node_container:
            st.image(st.session_state.results['node_visualization'], caption="Final Node Connections", use_container_width=True)
            
            # Display intermediate steps
            st.markdown("#### Intermediate Steps")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(st.session_state.results['node_mask'], caption="1. Emptied Mask")
            with col2:
                if st.session_state.results.get('enhanced_mask') is not None:
                    st.image(st.session_state.results['enhanced_mask'], caption="2. Enhanced Mask")
            with col3:
                st.image(st.session_state.results['contour_image'], caption="3. Contours")
            
            # Show corners image if available
            if st.session_state.results.get('corners_image') is not None:
                st.image(st.session_state.results['corners_image'], caption="4. Corner Detection", use_container_width=True)
    
    if st.button("Analyze Nodes"):
        print("Analyze Nodes button clicked")
        with st.spinner("Analyzing nodes..."):
            if st.session_state.results['bboxes'] is None:
                print("No bboxes found in session state")
                st.warning("Please detect components first")
            else:
                print(f"Found {len(st.session_state.results['bboxes'])} bboxes in session state")
                try:
                    
                    nodes, emptied_mask, enhanced, contour_image, corners_image, final_visualization = analyzer.get_node_connections(
                        st.session_state.results['original_image'], 
                        st.session_state.results['bboxes']
                    )
                    
                    st.session_state.results['nodes'] = nodes
                    st.session_state.results['node_visualization'] = final_visualization
                    st.session_state.results['node_mask'] = emptied_mask
                    st.session_state.results['enhanced_mask'] = enhanced
                    st.session_state.results['contour_image'] = contour_image
                    st.session_state.results['corners_image'] = corners_image
                    print(f"Node analysis complete. Found {len(nodes)} nodes")
                    
                    with node_container:
                        st.image(final_visualization, caption="Final Node Connections", use_container_width=True)
                        
                        # Display intermediate steps
                        st.markdown("#### Intermediate Steps")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.image(emptied_mask, caption="1. Emptied Mask")
                        with col2:
                            st.image(enhanced, caption="2. Enhanced Mask")
                        with col3:
                            st.image(contour_image, caption="3. Contours")
                        
                        # Show corners image
                        st.image(corners_image, caption="4. Corner Detection", use_container_width=True)
                except Exception as e:
                    print(f"Error during node analysis: {str(e)}")
                    st.error(f"Error during node analysis: {str(e)}")
                    import traceback
                    print("Full traceback:")
                    print(traceback.format_exc())

    # Step 4: Netlist Generation
    st.markdown("## Step 4: 📝 Netlist Generation")
    netlist_container = st.container()
    
    # Always show previous netlist results if they exist
    if st.session_state.results['netlist_text'] is not None:
        with netlist_container:
            st.markdown("### Initial Netlist")
            valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.results['nodes'])
            valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
            st.code(valueless_netlist_text, language="python")
            
            st.markdown("### Final Netlist with Component Values")
            st.code(st.session_state.results['netlist_text'], language="python")
    
    if st.button("Generate Netlist"):
        with st.spinner("Generating netlist..."):
            if st.session_state.results['nodes'] is not None:
                with netlist_container:
                    # Initial Netlist
                    st.markdown("### Initial Netlist")
                    valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.results['nodes'])
                    valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                    st.code(valueless_netlist_text, language="python")
                    
                    # Final Netlist
                    st.markdown("### Final Netlist with Component Values")
                    netlist = deepcopy(valueless_netlist)
                    enum_img, bbox_ids = analyzer.enumerate_components(st.session_state.results['original_image'], st.session_state.results['bboxes'])
                    
                    gemini_info = gemini_labels(enum_img)
                    analyzer.fix_netlist(netlist, gemini_info)
                    netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])
                    st.code(netlist_text, language="python")
                    
                    st.session_state.results['netlist'] = netlist
                    st.session_state.results['netlist_text'] = netlist_text
            else:
                st.warning("Please analyze nodes first")

    # Step 5: SPICE Analysis
    st.markdown("## Step 5: ⚡ SPICE Analysis")
    spice_container = st.container()
    if st.button("Run SPICE Analysis"):
        if st.session_state.results['netlist_text'] is not None:
            try:
                with spice_container:
                    net_text = '.title detected_circuit\n' + st.session_state.results['netlist_text']
                    parser = SpiceParser(source=net_text)
                    bootstrap_circuit = parser.build_circuit()
                    simulator = bootstrap_circuit.simulator()
                    analysis = simulator.operating_point()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Node Voltages")
                        st.json(analysis.nodes)
                    with col2:
                        st.markdown("### Branch Currents")
                        st.json(analysis.branches)
                
            except Exception as e:
                st.error(f"❌ SPICE Analysis Error: {str(e)}")
                st.info("💡 Tip: Check if all component values are properly detected and the circuit is properly connected.")
        else:
            st.warning("Please generate netlist first") 
```

