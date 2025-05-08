from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import cv2
import numpy as np
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from .utils import (
    load_classes,
    non_max_suppression_by_area
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
    device as sam2_device
)

class CircuitAnalyzer():
    def __init__(self, yolo_path='models/YOLO/best_large_model_yolo.pt', 
                 sam2_config_path='models/configs/sam2.1_hiera_l.yaml',
                 sam2_base_checkpoint_path='models/SAM2/sam2.1_hiera_large.pt',
                 sam2_finetuned_checkpoint_path='models/SAM2/best_miou_model_SAM_latest.pth',
                 use_sam2=True,
                 debug=False):
        self.yolo = YOLO(yolo_path)
        self.debug = debug
        self.classes = load_classes()
        self.classes_names = set(self.classes.keys())
        self.non_components = set(['text', 'junction', 'crossover', 'terminal', 'vss', 'explanatory', 'circuit', 'vss'])
        self.source_components = set(['voltage.ac', 'voltage.dc', 'voltage.battery', 'voltage.dependent', 'current.dc', 'current.dependent'])
        
        # Add property to store last SAM2 output
        self.last_sam2_output = None
        
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
                self.sam2_device = sam2_device
                print(f"Using SAM2 device: {self.sam2_device}")
                
                # Define LoRA target modules
                base_parts = [
                    "sam_mask_decoder.transformer.layers.0.self_attn.k_proj",
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
                    "sam_mask_decoder.transformer.layers.1.mlp.layers.1"
                    ]
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
                    "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj",
                    ]
                
                # Build Model Structure using the provided paths
                print(f"Building SAM2 model with config: {sam2_config_path} and base checkpoint: {sam2_base_checkpoint_path}")
                self.sam2_model = get_modified_sam2(
                    model_cfg_path=f"/{sam2_config_path}" if not sam2_config_path.startswith('/') else sam2_config_path,
                    checkpoint_path=str(sam2_base_checkpoint_path),
                    device=str(self.sam2_device),
                    use_high_res_features=True,
                    use_peft=True,
                    lora_rank=4,
                    lora_alpha=16,
                    lora_dropout=0.3,
                    lora_target_modules=base_parts + added_parts,
                    use_wrapper=True,
                    trainable_embedding_r=4,
                    use_refinement_layer=True,
                    refinement_kernels=[3, 5, 7, 11],
                    kernel_channels=2,
                    weight_dice=0.5, 
                    weight_focal=0.4, 
                    weight_iou=0.3, 
                    weight_freq=0.1,
                    focal_alpha=0.25
                )
                
                # Load the fine-tuned weights
                print(f"Loading SAM2 fine-tuned weights from: {sam2_finetuned_checkpoint_path}")
                checkpoint = torch.load(str(sam2_finetuned_checkpoint_path), map_location=self.sam2_device)
                if 'state_dict' in checkpoint:
                    model_state_dict = checkpoint['state_dict']
                else:
                    model_state_dict = checkpoint
                    
                self.sam2_model.load_state_dict(model_state_dict)
                self.sam2_model.eval()  # Set to evaluation mode
                
                # Initialize transforms
                if hasattr(self.sam2_model, 'sam2_model') and hasattr(self.sam2_model.sam2_model, 'image_size'):
                    resolution = self.sam2_model.sam2_model.image_size
                elif hasattr(self.sam2_model, 'image_size'):
                    resolution = self.sam2_model.image_size
                else:
                    resolution = 1024  # Default fallback
                    print(f"Warning: Could not determine SAM2 model image size, using default: {resolution}")
                
                self.sam2_transforms = SAM2Transforms(
                    resolution=resolution,
                    mask_threshold=0,
                    max_hole_area=0,
                    max_sprinkle_area=0
                )
                
                print("SAM2 Model loaded successfully")
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
        boxes = [{'class': img_classes[i], 
                  'confidence': confidences[i], 
                  'xmin': round(xmin), 
                  'ymin': round(ymin), 
                  'xmax': round(xmax), 
                  'ymax': round(ymax), 
                  'persistent_uid': f"{img_classes[i]}_{round(xmin)}_{round(ymin)}_{round(xmax)}_{round(ymax)}"} 
                 for i, (xmin, ymin, xmax, ymax) in enumerate(boxes)]
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
        Also computes a bounding box encompassing all detected SAM2 mask contours.

        Args:
            image_np_bgr (np.ndarray): Input image in BGR format (from cv2.imread).

        Returns:
            Tuple[np.ndarray | None, np.ndarray | None, Tuple[int,int,int,int] | None]:
                - mask_binary_np: Binary mask (uint8, 0 or 255) from SAM2, at original image resolution.
                - self.last_sam2_output: Colored version of the mask for display.
                - sam_extent_bbox: Tuple (xmin, ymin, xmax, ymax) of the bounding box around SAM2 contours, or None.
        """
        if not self.use_sam2 or self.sam2_model is None or self.sam2_transforms is None:
            print("SAM 2 is not available or not initialized. Cannot segment.")
            self.last_sam2_output = None
            return None, None, None

        if self.debug:
            print("Segmenting with SAM 2...")
        try:
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_np_rgb)
            orig_hw = image_np_rgb.shape[:2]

            image_tensor = self.sam2_transforms(image_pil).unsqueeze(0).to(self.sam2_device)

            self.sam2_model.eval()
            with torch.no_grad():
                high_res_mask, low_res_mask, _ = self.sam2_model(image_tensor)
                chosen_mask = high_res_mask

            final_mask_tensor = self.sam2_transforms.postprocess_masks(chosen_mask, orig_hw)
            mask_squeezed = final_mask_tensor.detach().cpu().squeeze()
            mask_binary_np = (mask_squeezed > 0.0).numpy().astype(np.uint8) * 255

            # Store colored version for display
            colored_output = cv2.cvtColor(mask_binary_np, cv2.COLOR_GRAY2BGR)
            colored_output[:,:,0] = 0
            colored_output[:,:,2] = 0
            self.last_sam2_output = colored_output

            # --- Calculate bounding box of SAM2 mask contours ---
            sam_extent_bbox = None
            contours_sam, _ = cv2.findContours(mask_binary_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_sam:
                all_points = np.concatenate(contours_sam)
                x_sam, y_sam, w_sam, h_sam = cv2.boundingRect(all_points)
                sam_extent_bbox = (x_sam, y_sam, x_sam + w_sam, y_sam + h_sam)
                if self.debug:
                    print(f"SAM2 Extent BBox calculated: {sam_extent_bbox}")
            elif self.debug:
                print("No contours found in SAM2 mask to calculate extent bbox.")
            # --- End BBox Calculation ---

            if self.debug:
                print("SAM 2 Segmentation successful.")
            return mask_binary_np, self.last_sam2_output, sam_extent_bbox

        except Exception as e:
            print(f"Error during SAM 2 segmentation: {e}")
            import traceback
            print(traceback.format_exc())
            self.last_sam2_output = None
            return None, None, None
    
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
        resized_bboxes = []
        for bbox in bboxes:
            resized_bbox = bbox.copy() # Start by copying all original key-value pairs
            
            xmin_resized = int(bbox['xmin'] * width_scale)
            ymin_resized = int(bbox['ymin'] * height_scale)
            xmax_resized = int(bbox['xmax'] * width_scale)
            ymax_resized = int(bbox['ymax'] * height_scale)

            resized_bbox['xmin'] = xmin_resized
            resized_bbox['ymin'] = ymin_resized
            resized_bbox['xmax'] = xmax_resized
            resized_bbox['ymax'] = ymax_resized
            
            resized_bboxes.append(resized_bbox)
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
    
    
    def crop_image_and_adjust_bboxes(self, image_to_crop, bboxes_to_adjust, crop_defining_bbox, padding=10):
        """
        Crops an image based on a defining bounding box and adjusts other bounding boxes.
        Preserves persistent_uid on adjusted bboxes.

        Args:
            image_to_crop (np.ndarray): The original image to be cropped.
            bboxes_to_adjust (List[Dict]): List of bboxes (e.g., from YOLO) in original image coordinates.
            crop_defining_bbox (Tuple[int,int,int,int] | None): Bbox (xmin, ymin, xmax, ymax) defining the crop area.
                                                        If None, returns originals.
            padding (int): Padding around crop_defining_bbox.

        Returns:
            Tuple[np.ndarray, List[Dict], Tuple[int, int, int, int] | None]:
                - Cropped image (or original if no crop_defining_bbox or invalid crop).
                - New list of adjusted bboxes (or original if no crop).
                - Crop area details (crop_abs_xmin, crop_abs_ymin, new_width, new_height) relative to original, or None.
        """
        if crop_defining_bbox is None:
            if self.debug:
                print("Crop: No crop_defining_bbox provided. Returning originals.")
            return image_to_crop, [deepcopy(b) for b in bboxes_to_adjust], None

        original_height, original_width = image_to_crop.shape[:2]
        def_xmin, def_ymin, def_xmax, def_ymax = crop_defining_bbox

        crop_abs_xmin = max(0, def_xmin - padding)
        crop_abs_ymin = max(0, def_ymin - padding)
        crop_abs_xmax = min(original_width, def_xmax + padding)
        crop_abs_ymax = min(original_height, def_ymax + padding)

        if self.debug:
            print(f"Crop: Original dims: {original_width}x{original_height}")
            print(f"Crop: Defining bbox for crop: {crop_defining_bbox}")
            print(f"Crop: Calculated absolute crop area (xmin,ymin,xmax,ymax): {crop_abs_xmin}, {crop_abs_ymin}, {crop_abs_xmax}, {crop_abs_ymax}")

        if crop_abs_xmin >= crop_abs_xmax or crop_abs_ymin >= crop_abs_ymax:
            if self.debug:
                print("Crop: Invalid crop region. Returning originals.")
            return image_to_crop, [deepcopy(b) for b in bboxes_to_adjust], None

        cropped_image = image_to_crop[crop_abs_ymin:crop_abs_ymax, crop_abs_xmin:crop_abs_xmax]
        new_height, new_width = cropped_image.shape[:2]
        if self.debug:
            print(f"Crop: Cropped image shape: {cropped_image.shape}")

        adjusted_bboxes = []
        for bbox_original in bboxes_to_adjust:
            adj_bbox = deepcopy(bbox_original) # Carries over persistent_uid
            adj_bbox['xmin'] = bbox_original['xmin'] - crop_abs_xmin
            adj_bbox['ymin'] = bbox_original['ymin'] - crop_abs_ymin
            adj_bbox['xmax'] = bbox_original['xmax'] - crop_abs_xmin
            adj_bbox['ymax'] = bbox_original['ymax'] - crop_abs_ymin

            adj_bbox['xmin'] = max(0, adj_bbox['xmin'])
            adj_bbox['ymin'] = max(0, adj_bbox['ymin'])
            adj_bbox['xmax'] = min(new_width, adj_bbox['xmax'])
            adj_bbox['ymax'] = min(new_height, adj_bbox['ymax'])

            if adj_bbox['xmax'] > adj_bbox['xmin'] and adj_bbox['ymax'] > adj_bbox['ymin']:
                adjusted_bboxes.append(adj_bbox)
            elif self.debug:
                print(f"Crop: Bbox for class {adj_bbox.get('class')} (UID: {adj_bbox.get('persistent_uid')}) filtered out. Orig coords: {bbox_original[ 'xmin']},{bbox_original['ymin']}. Adjusted: {adj_bbox[ 'xmin']},{adj_bbox['ymin']}")
        
        if self.debug:
            print(f"Crop: Original bbox count: {len(bboxes_to_adjust)}, Adjusted bbox count: {len(adjusted_bboxes)}")
        
        return cropped_image, adjusted_bboxes, (crop_abs_xmin, crop_abs_ymin, new_width, new_height)

    def get_node_connections(self, _image_for_context, processing_wire_mask, bboxes_relative_to_mask):
        if self.debug:
            print(f"Node Connections: Received processing_wire_mask shape: {processing_wire_mask.shape}, {len(bboxes_relative_to_mask)} bboxes.")

        # --- Mask Preparation (Input `processing_wire_mask` is already the cropped SAM2 mask) ---
        emptied_mask = processing_wire_mask.copy()
        current_height, current_width = emptied_mask.shape[:2]

        for bbox_comp in bboxes_relative_to_mask:
            if bbox_comp['class'] not in ('crossover', 'junction', 'terminal', 'circuit', 'vss'): # 'circuit' may not exist or be relevant here
                ymin, ymax = max(0, int(bbox_comp['ymin'])), min(current_height, int(bbox_comp['ymax']))
                xmin, xmax = max(0, int(bbox_comp['xmin'])), min(current_width, int(bbox_comp['xmax']))
                if ymin < ymax and xmin < xmax:
                    emptied_mask[ymin:ymax, xmin:xmax] = 0
        
        if self.debug:
            self.show_image(emptied_mask, 'Emptied Mask (from cropped SAM2, pre-resize)')

        # Resize this emptied_mask (derived from cropped SAM2) and its corresponding bboxes for contour processing
        # Note: bboxes_relative_to_mask are already relative to the full extent of emptied_mask before this resize.
        processing_mask_resized, processing_bboxes_resized = self.resize_image_keep_aspect(emptied_mask, bboxes_relative_to_mask)

        if self.debug:
            self.show_image(processing_mask_resized, 'Processing Mask (Resized Emptied)')

        enhanced = self.enhance_lines(processing_mask_resized)
        if self.debug:
            self.show_image(enhanced, 'Enhanced Processing Mask')
            
        contours, contour_image_viz = self.get_contours(enhanced)
        if self.debug:
            self.show_image(contour_image_viz, 'Contours on Resized Processing Mask')
            
        nodes = {item['id']:{'id': item['id'], 'components': [], 'contour': item['contour']} for item in contours}
        
        rects_image_viz = contour_image_viz.copy()

        # Loop through the bboxes that were resized along with the processing_mask_resized
        for i, bbox_comp_proc_resized in enumerate(processing_bboxes_resized):
            if bbox_comp_proc_resized['class'] in self.non_components:
                continue
            
            if self.debug: 
                cv2.rectangle(rects_image_viz, 
                              (int(bbox_comp_proc_resized['xmin']), int(bbox_comp_proc_resized['ymin'])), 
                              (int(bbox_comp_proc_resized['xmax']), int(bbox_comp_proc_resized['ymax'])), 
                              (255, 100, 100), 1)

            for contour_item in contours:
                c_xmin_contour, c_ymin_contour, c_width_contour, c_height_contour = contour_item['rectangle']
                c_xmax_contour = c_xmin_contour + c_width_contour
                c_ymax_contour = c_ymin_contour + c_height_contour
                
                if self.debug: 
                     cv2.rectangle(rects_image_viz, (c_xmin_contour, c_ymin_contour), (c_xmax_contour, c_ymax_contour), color=(0, 255, 0), thickness=1)
    
                # Broad phase using resized coordinates
                if bbox_comp_proc_resized['xmax'] < c_xmin_contour or bbox_comp_proc_resized['xmin'] > c_xmax_contour or \
                   bbox_comp_proc_resized['ymax'] < c_ymin_contour or bbox_comp_proc_resized['ymin'] > c_ymax_contour:
                    continue
                
                for point_array in contour_item['contour']:
                    point = tuple(point_array[0]) # Point is in resized coordinate system
                    
                    # is_point_near_bbox expects bbox_comp_proc_resized (already in resized system)
                    if self.is_point_near_bbox(point, bbox_comp_proc_resized, pixel_threshold=6):
                        # IMPORTANT: Store the component from bboxes_relative_to_mask (original scale of the cropped image),
                        # using the index `i` from the enumerated processing_bboxes_resized.
                        # This ensures persistent_uid and correct coordinates for fix_netlist are stored.
                        target_bbox_to_add = deepcopy(bboxes_relative_to_mask[i])
                        
                        component_unique_ref = target_bbox_to_add.get('persistent_uid')
                        # Fallback if persistent_uid is somehow missing (should not happen with current crop logic)
                        if component_unique_ref is None: 
                             if self.debug: print(f"WARNING: persistent_uid missing for component: {target_bbox_to_add.get('class')}")
                             component_unique_ref = (target_bbox_to_add['class'], target_bbox_to_add['xmin'], target_bbox_to_add['ymin'], target_bbox_to_add['xmax'], target_bbox_to_add['ymax'])

                        is_already_added = False
                        for existing_comp_node_data in nodes[contour_item['id']]['components']:
                            existing_comp_ref = existing_comp_node_data.get('persistent_uid')
                            if existing_comp_ref is None: # Fallback for existing components in node if UID was missing
                                existing_comp_ref = (existing_comp_node_data['class'], existing_comp_node_data['xmin'], existing_comp_node_data['ymin'], existing_comp_node_data['xmax'], existing_comp_node_data['ymax'])
                            
                            if existing_comp_ref == component_unique_ref:
                                is_already_added = True
                                break
                        
                        if not is_already_added:
                            nodes[contour_item['id']]['components'].append(target_bbox_to_add)
                            if self.debug:
                                print(f"Node Conn: Added comp UID {target_bbox_to_add.get('persistent_uid')} to node {contour_item['id']}")
                        break # Found connection for this contour and component
        
        if self.debug:
            self.show_image(rects_image_viz, "Nodes - Contour/ResizedProcBBox Overlaps")

        valid_nodes = {node_id: node_data for node_id, node_data in nodes.items() if node_data['components']}
        
        if not valid_nodes: 
            if self.debug:
                print("Node Conn: No valid nodes with components found.")
            # Return structures matching expected output, but empty/default where appropriate
            # final_node_viz_image should be based on processing_mask_resized dimensions
            viz_fallback = processing_mask_resized.copy()
            if len(viz_fallback.shape) == 2: viz_fallback = cv2.cvtColor(viz_fallback, cv2.COLOR_GRAY2BGR)
            return [], emptied_mask, enhanced, contour_image_viz, viz_fallback

        max_connections_node_val = max(len(node_data['components']) for node_data in valid_nodes.values())
        
        nodes_with_max = [node_id for node_id, node_data in valid_nodes.items() 
                         if len(node_data['components']) == max_connections_node_val]

        chosen_ground_node_old_id = None
        if len(nodes_with_max) == 1:
            chosen_ground_node_old_id = nodes_with_max[0]
        else: 
            source_connected_nodes_for_ground = []
            # Check against bboxes_relative_to_mask for source components as they have original class names
            for node_id_candidate in nodes_with_max:
                # Components stored in valid_nodes are from bboxes_relative_to_mask, so they have correct class for source check
                components_in_node = valid_nodes[node_id_candidate]['components']
                if any(comp['class'] in self.source_components for comp in components_in_node):
                    contour_detail_for_node = next((c for c in contours if c['id'] == node_id_candidate), None)
                    if contour_detail_for_node:
                         # Use centroid y of contour_detail_for_node (which is in resized space)
                         rect_x, rect_y, rect_w, rect_h = contour_detail_for_node['rectangle']
                         source_connected_nodes_for_ground.append((node_id_candidate, rect_y + rect_h / 2))
            
            if source_connected_nodes_for_ground:
                source_connected_nodes_for_ground.sort(key=lambda x: -x[1]) # Higher Y in image (lower on schematic) is preferred for ground
                chosen_ground_node_old_id = source_connected_nodes_for_ground[0][0]
            elif nodes_with_max: # If no sources, but still candidates for ground
                chosen_ground_node_old_id = nodes_with_max[0]
            else: # Should not be reached if valid_nodes is not empty
                chosen_ground_node_old_id = list(valid_nodes.keys())[0] if valid_nodes else None 
        
        new_nodes_list = []
        if chosen_ground_node_old_id is not None and chosen_ground_node_old_id in valid_nodes:
            new_nodes_list.append({
                'id': 0, 
                'components': valid_nodes[chosen_ground_node_old_id]['components'],
                'contour': valid_nodes[chosen_ground_node_old_id]['contour'] # Contour is from resized space
            })
            
            next_node_id_val = 1
            # Sort other nodes by their original ID for consistent numbering if possible
            sorted_other_node_ids = sorted([nid for nid in valid_nodes.keys() if nid != chosen_ground_node_old_id])

            for old_node_id in sorted_other_node_ids:
                node_data = valid_nodes[old_node_id]
                # Ensure node has at least 2 components, unless it's the only other node making it node 1
                if len(node_data['components']) >= 2 or (len(new_nodes_list) == 1 and len(valid_nodes) == 2 and len(node_data['components']) > 0 ) :
                    new_nodes_list.append({
                        'id': next_node_id_val,
                        'components': node_data['components'],
                        'contour': node_data['contour'] # Contour is from resized space
                    })
                    next_node_id_val += 1
        # Fallback if ground node wasn't determined (should be rare if valid_nodes exist)
        elif valid_nodes: 
            if self.debug: print("Node Conn: Ground node ID not properly determined. Arbitrary numbering for all valid nodes.")
            next_node_id_val = 0
            for old_node_id in sorted(valid_nodes.keys()): # Iterate through all valid nodes
                node_data = valid_nodes[old_node_id]
                if len(node_data['components']) > 0: # Take any node with components
                    new_nodes_list.append({
                        'id': next_node_id_val,
                        'components': node_data['components'],
                        'contour': node_data['contour']
                    })
                    next_node_id_val += 1

        # Final visualization should be on processing_mask_resized dimensions
        final_node_viz_image = processing_mask_resized.copy()
        if len(final_node_viz_image.shape) == 2: 
            final_node_viz_image = cv2.cvtColor(final_node_viz_image, cv2.COLOR_GRAY2BGR)
        
        for node_item_final in new_nodes_list:
            # Moments and drawing are on contours from processing_mask_resized
            M = cv2.moments(node_item_final['contour'])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(final_node_viz_image, [node_item_final['contour']], -1, (0, 255, 0), 2) 
                cv2.putText(final_node_viz_image, str(node_item_final['id']), 
                            (cx-10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 0, 255), 2) 

        return new_nodes_list, emptied_mask, enhanced, contour_image_viz, final_node_viz_image

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
                
                if self.debug:
                    print(f"Debug generate_netlist: Processing component for line update: {component}")
                    if 'persistent_uid' not in component:
                        print(f"Debug generate_netlist: persistent_uid MISSING from component dict before update: {component}")

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
                if self.debug and 'persistent_uid' not in line:
                    print(f"Debug generate_netlist: persistent_uid MISSING from line dict AFTER update. Original component was: {component}")
                netlist.append(line)
        
        return netlist

    def fix_netlist(self, netlist, vlm_out, all_enumerated_bboxes):
        for line_from_gen_netlist in netlist:
            target_persistent_uid = line_from_gen_netlist.get('persistent_uid')
            if not target_persistent_uid:
                if self.debug:
                    print(f"Debug: Line in netlist missing persistent_uid: {line_from_gen_netlist}")
                continue

            visual_id_for_this_component = None
            # Find the visual ID associated with this persistent_uid from all_enumerated_bboxes
            for enum_bbox in all_enumerated_bboxes:
                if enum_bbox.get('persistent_uid') == target_persistent_uid:
                    visual_id_for_this_component = enum_bbox.get('id') # This 'id' is the visual enum
                    break
            
            if visual_id_for_this_component is None:
                if self.debug:
                    print(f"Debug: Could not find visual_id for persistent_uid {target_persistent_uid}")
                continue

            # Now find the item in vlm_out (Gemini's output) that has this visual_id
            for vlm_item in vlm_out:
                if str(vlm_item.get('id')) == str(visual_id_for_this_component):
                    # Apply updates from vlm_item to line_from_gen_netlist
                    current_value_in_netlist_line = line_from_gen_netlist.get('value') # Initially the string "None"
                    vlm_provided_value = vlm_item.get('value') # Value from Gemini, e.g., "2k" or Python None

                    # Potentially override vlm_provided_value if it's problematic for an independent source
                    effective_vlm_value = vlm_provided_value
                    # Check the component_type *after* potential update from VLM class below
                    # We need to know the final intended type of the component.
                    
                    # Store original class and type before VLM override for comparison
                    original_yolo_class = line_from_gen_netlist.get('class')
                    original_component_type = line_from_gen_netlist.get('component_type')
                    
                    vlm_class = vlm_item.get('class')
                    prospective_component_type = self.netlist_map.get(vlm_class, 'UN') # Get component type based on VLM's class

                    if prospective_component_type in ['V', 'I']: # Independent V or I source based on VLM's classification
                        if isinstance(vlm_provided_value, str):
                            try:
                                float(vlm_provided_value) # Check if it's a number string
                                # If successful, it's a numeric string, so it's a valid value.
                            except ValueError:
                                # It's a string but not a simple number.
                                # If it's purely alphabetical (e.g., "ia") and not 'ac' (valid for AC sources), treat as invalid.
                                if vlm_provided_value.isalpha() and vlm_provided_value.lower() != 'ac':
                                    if self.debug:
                                        print(f"Debug fix_netlist: Independent source type {prospective_component_type} "
                                              f"derived from VLM class {vlm_class} with problematic alpha value '{vlm_provided_value}'. Setting effective value to None.")
                                    effective_vlm_value = None
                    
                    if self.debug:
                        print(f"Debug fix_netlist: comp_uid={target_persistent_uid}, visual_id={visual_id_for_this_component}, vlm_item_id={vlm_item.get('id')}")
                        print(f"Debug fix_netlist: current_value_in_netlist_line='{current_value_in_netlist_line}' (type: {type(current_value_in_netlist_line)})")
                        print(f"Debug fix_netlist: vlm_provided_value='{vlm_provided_value}' (type: {type(vlm_provided_value)})")
                        print(f"Debug fix_netlist: prospective_component_type='{prospective_component_type}', effective_vlm_value='{effective_vlm_value}' (type: {type(effective_vlm_value)})")

                    if current_value_in_netlist_line is None or str(current_value_in_netlist_line).strip().lower() == 'none':
                        line_from_gen_netlist['value'] = effective_vlm_value # Use potentially modified value
                        if self.debug:
                            print(f"Debug fix_netlist: Value UPDATED to '{line_from_gen_netlist.get('value')}' (type: {type(line_from_gen_netlist.get('value'))})")
                    # If current value was not None, but our new logic sets effective_vlm_value to None (for problematic V/I sources), we should update
                    elif effective_vlm_value is None and \
                         prospective_component_type in ['V', 'I'] and \
                         (current_value_in_netlist_line is not None and str(current_value_in_netlist_line).strip().lower() != 'none'):
                        line_from_gen_netlist['value'] = None
                        if self.debug:
                            print(f"Debug fix_netlist: Value OVERRIDDEN to None for {prospective_component_type} "
                                  f"due to problematic VLM value '{vlm_provided_value}'. Original line value: '{current_value_in_netlist_line}'")
                    else:
                        if self.debug:
                            print(f"Debug fix_netlist: Value NOT updated, current_value_in_netlist_line was '{current_value_in_netlist_line}'")

                    # Update class and component_type based on VLM, this should happen *after* value check related to prospective_component_type
                    if original_yolo_class != vlm_class: # Compare with original YOLO class
                        if self.debug:
                            print(f"Fixing Netlist: Component with p_uid {target_persistent_uid} (VisualID {visual_id_for_this_component}). "
                                  f"Original YOLO class: {original_yolo_class}, "
                                  f"VLM class: {vlm_class}")
                        
                        line_from_gen_netlist['class'] = vlm_class
                        line_from_gen_netlist['component_type'] = prospective_component_type # Use the already determined type
                        
                        # Recalculate component_num if the component_type prefix changes
                        # or if the class changes significantly.
                        if original_component_type != prospective_component_type or original_yolo_class != vlm_class :
                            new_class_component_numbers = []
                            for l_other in netlist:
                                if l_other.get('persistent_uid') != target_persistent_uid and l_other.get('class') == vlm_class:
                                    num = l_other.get('component_num')
                                    if num is not None:
                                        try:
                                            new_class_component_numbers.append(int(num))
                                        except ValueError:
                                            if self.debug:
                                                print(f"Debug: Non-integer component_num {num} for class {vlm_class}")
                            line_from_gen_netlist['component_num'] = max(new_class_component_numbers, default=0) + 1


                    if vlm_class == 'gnd':
                        line_from_gen_netlist['node_2'] = 0
                    
                    break # Found and processed vlm_item for this line_from_gen_netlist
        # Potentially add a global re-numbering pass here if strict R1,R2, C1,C2 ordering is critical across all components.

    def stringify_line(self, netlist_line):
        if netlist_line.get('class') == 'gnd':
            return ""  # Ground components don't have a direct SPICE line; their nodes become '0'
        return f"{netlist_line['component_type']}{netlist_line['component_num']} {netlist_line['node_1']} {netlist_line['node_2']} {netlist_line['value']}"