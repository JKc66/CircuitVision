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

# +++ Imports for Groq/LLaMA Direction Analysis +++
import base64
import io
from PIL import Image as PILImage # Alias to avoid conflict with cv2.Image
from dotenv import load_dotenv
from groq import Groq
import time
import json # For parsing LLaMA JSON output

# Load environment variables (e.g., for GROQ_API_KEY)
load_dotenv()
# --- End Imports for Groq/LLaMA ---

class CircuitAnalyzer():
    def __init__(self, yolo_path='models/YOLO/best_large_model_yolo.pt', 
                 sam2_config_path='models/configs/sam2.1_hiera_l.yaml',
                 sam2_base_checkpoint_path='models/SAM2/sam2.1_hiera_large.pt',
                 sam2_finetuned_checkpoint_path='models/SAM2/best_miou_model_SAM_latest.pth',
                 use_sam2=True,
                 debug=False):
        self.yolo = YOLO(yolo_path)
        self.debug = debug
        if self.debug:
            self.last_llama_input_images = {}
        self.classes = load_classes()
        self.classes_names = set(self.classes.keys())
        self.non_components = set(['text', 'junction', 'crossover', 'terminal', 'vss', 'explanatory', 'circuit', 'vss'])
        self.source_components = set(['voltage.ac', 'voltage.dc', 'voltage.dependent', 'current.dc', 'current.dependent'])
        
        # Add property to store last SAM2 output
        self.last_sam2_output = None
        
        problematic = set(['__background__', 'inductor.coupled', 'operational_amplifier.schmitt_trigger', 'mechanical', 'optical', 'block', 'magnetic', 'antenna', 'relay'])
        reducing = set(['operational_amplifier.schmitt_trigger', 'integrated_circuit.ne555', 'resistor.photo', 'diode.thyrector'])
        deleting = set(['optical', '__background__', 'inductor.coupled', 'mechanical', 'block','magnetic'])
        unknown = set(['relay','antenna','diac','triac', 'crystal','antenna', 'probe', 'probe.current', 'probe.voltage', 'optocoupler', 'socket', 'fuse', 'speaker', 'motor', 'lamp', 'microphone','transistor.photo','xor', 'and', 'or', 'not', 'nand', 'nor'])
        
        self.classes_names = self.classes_names - deleting - unknown - reducing
        self.classes = {key: value for key, value in self.classes.items() if key in self.classes_names}
        self.classes = {key: i for i, key, in enumerate(self.classes.keys())}
        
        self.project_classes = set(['gnd', 'voltage.ac', 'voltage.dc', 'resistor', 'voltage.dependent', 'current.dc', 'current.dependent', 'capacitor', 'inductor', 'diode'])
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
            'current.ac': 'I',  
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

        # +++ Component types for semantic direction analysis +++
        # These are string names. The numeric IDs will be used in the enrichment function.
        self.yolo_class_names_map = self.yolo.model.names # Get mapping from class ID to name from YOLO model
        self.classes_of_interest_names = {
            'voltage.dc', 'voltage.ac', 
            'diode', 'diode.light_emitting', 'diode.zener',
            'transistor.bjt', 
            'unknown' 
        }
        self.voltage_classes_names = {'voltage.dc', 'voltage.ac', 'transistor.bjt', 'unknown'}
        self.diode_classes_names = {'diode', 'diode.light_emitting', 'diode.zener'}
        self.current_source_classes_names = {'current.dc', 'current.dependent'}
        # --- End component types ---

        # +++ Groq Client Initialization +++
        self.groq_client = None
        if os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                if self.debug:
                    print("Groq client initialized successfully.")
                    # It makes sense to initialize last_llama_input_images here as well, 
                    # or ensure it's initialized if debug is true, as it's related to LLaMA/Groq debugging.
                    if not hasattr(self, 'last_llama_input_images'): # Ensure it's initialized if not already
                        self.last_llama_input_images = {}
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
        elif self.debug:
            print("GROQ_API_KEY not found, semantic direction analysis via LLaMA will be disabled.")
        # ---

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
        img_classes_ids = results.boxes.cls.cpu().numpy().tolist() # Get numeric class IDs
        img_classes_names = [results.names[int(num_id)] for num_id in img_classes_ids] # Get string names
        confidences = results.boxes.conf.cpu().numpy().tolist()
        boxes_coords = results.boxes.xyxy.cpu().numpy().tolist()
        
        processed_boxes = []
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes_coords):
            processed_boxes.append({
                'class': img_classes_names[i],
                '_yolo_class_id_temp': int(img_classes_ids[i]), # Store numeric ID temporarily
                'confidence': confidences[i],
                'xmin': round(xmin),
                'ymin': round(ymin),
                'xmax': round(xmax),
                'ymax': round(ymax),
                'persistent_uid': f"{img_classes_names[i]}_{round(xmin)}_{round(ymin)}_{round(xmax)}_{round(ymax)}"
            })
        return processed_boxes

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

        # Define a palette of bright colors
        bright_colors_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Lime
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Violet
            (0, 255, 128),  # Spring Green
            (255, 192, 203), # Pink
            (173, 216, 230), # Light Blue
            (255, 165, 0),  # Orange (another shade)
            (127, 255, 212), # Aquamarine
            (240, 230, 140), # Khaki (light enough)
            (255, 105, 180)  # Hot Pink
        ]

        # Iterate through the contours and visualize each
        for contour_item in contours: # Renamed loop variable for clarity
            # Choose a color from the palette
            color = bright_colors_palette[contour_item['id'] % len(bright_colors_palette)]
            cv2.drawContours(contour_img, [contour_item['contour']], -1, color, 2)

            font=cv2.FONT_HERSHEY_SIMPLEX
            font_scale=0.5
            # Text color for contour ID is Red
            text_color=(255, 0, 0) 
            thickness=2

            # Calculate the contour's centroid
            M = cv2.moments(contour_item['contour'])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0 

            # Get the text for the current contour
            text = str(contour_item['id'])

            # Put the text on the image
            cv2.putText(contour_img, text, (cX + 10, cY + 10), 
                        font, font_scale, text_color, thickness)
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
            Tuple[np.ndarray, List[Dict], Dict | None]:
                - Cropped image (or original if no crop_defining_bbox or invalid crop).
                - New list of adjusted bboxes (or original if no crop).
                - crop_debug_info (Dict): A dictionary containing detailed information about the crop operation, 
                                          or None if no crop_defining_bbox was provided.
        """
        crop_debug_info = {
            'crop_applied': False,
            'reason_for_no_crop': None,
            'original_image_dims': image_to_crop.shape[:2][::-1], # (width, height)
            'defining_bbox': None,
            'padding_value': padding,
            'initial_calculated_window_before_text': None,
            'text_bboxes_that_expanded_crop': [],
            'final_crop_window_abs': None,
            'cropped_image_dims': None
        }

        if crop_defining_bbox is None:
            if self.debug:
                print("Crop: No crop_defining_bbox provided. Returning originals.")
            crop_debug_info['reason_for_no_crop'] = "no_defining_bbox"
            return image_to_crop, [deepcopy(b) for b in bboxes_to_adjust], crop_debug_info

        original_height, original_width = image_to_crop.shape[:2]
        def_xmin, def_ymin, def_xmax, def_ymax = crop_defining_bbox
        crop_debug_info['defining_bbox'] = (def_xmin, def_ymin, def_xmax, def_ymax)

        # Early exit if crop_defining_bbox is already large relative to the image
        original_area = float(original_height * original_width)
        crop_defining_bbox_width = float(max(0, def_xmax - def_xmin))
        crop_defining_bbox_height = float(max(0, def_ymax - def_ymin))
        crop_defining_bbox_area = crop_defining_bbox_width * crop_defining_bbox_height

        if original_area > 0 and (crop_defining_bbox_area / original_area) > 0.75: # Threshold: 75%
            if self.debug:
                print(f"Crop: crop_defining_bbox area ({crop_defining_bbox_area:.0f}) is > 75% of original image area ({original_area:.0f}). Skipping crop.")
            crop_debug_info['reason_for_no_crop'] = "focused_image_skip"
            crop_debug_info['cropped_image_dims'] = crop_debug_info['original_image_dims'] # No change
            return image_to_crop, [deepcopy(b) for b in bboxes_to_adjust], crop_debug_info

        # Initial crop window calculation based on crop_defining_bbox and padding
        current_crop_xmin = float(max(0, def_xmin - padding))
        current_crop_ymin = float(max(0, def_ymin - padding))
        current_crop_xmax = float(min(original_width, def_xmax + padding))
        current_crop_ymax = float(min(original_height, def_ymax + padding))
        crop_debug_info['initial_calculated_window_before_text'] = (int(round(current_crop_xmin)), int(round(current_crop_ymin)), int(round(current_crop_xmax)), int(round(current_crop_ymax)))
        
        if self.debug:
            print(f"Crop: Initial calc. crop window (SAM extent + padding): x({current_crop_xmin:.0f}-{current_crop_xmax:.0f}), y({current_crop_ymin:.0f}-{current_crop_ymax:.0f})")

        # Expand crop window to include nearby 'text' boxes
        text_inclusion_padding = 15

        for bbox_original in bboxes_to_adjust:
            if bbox_original.get('class') == 'text':
                text_xmin, text_ymin, text_xmax, text_ymax = float(bbox_original['xmin']), float(bbox_original['ymin']), float(bbox_original['xmax']), float(bbox_original['ymax'])
                interest_area_xmin = float(def_xmin - padding)
                interest_area_ymin = float(def_ymin - padding)
                interest_area_xmax = float(def_xmax + padding)
                interest_area_ymax = float(def_ymax + padding)
                
                overlaps_interest_area = not (text_xmax < interest_area_xmin or \
                                              text_xmin > interest_area_xmax or \
                                              text_ymax < interest_area_ymin or \
                                              text_ymin > interest_area_ymax)

                if overlaps_interest_area:
                    expanded_xmin = min(current_crop_xmin, max(0, text_xmin - text_inclusion_padding))
                    expanded_ymin = min(current_crop_ymin, max(0, text_ymin - text_inclusion_padding))
                    expanded_xmax = max(current_crop_xmax, min(original_width, text_xmax + text_inclusion_padding))
                    expanded_ymax = max(current_crop_ymax, min(original_height, text_ymax + text_inclusion_padding))

                    # Check if this text box actually caused an expansion
                    did_expand = (expanded_xmin != current_crop_xmin or 
                                  expanded_ymin != current_crop_ymin or 
                                  expanded_xmax != current_crop_xmax or 
                                  expanded_ymax != current_crop_ymax)

                    current_crop_xmin, current_crop_ymin, current_crop_xmax, current_crop_ymax = expanded_xmin, expanded_ymin, expanded_xmax, expanded_ymax
                    
                    if did_expand:
                        if self.debug:
                            print(f"Crop: Text box UID {bbox_original.get('persistent_uid')} at ({text_xmin:.0f},{text_ymin:.0f})-({text_xmax:.0f},{text_ymax:.0f}) caused expansion.")
                            print(f"Crop: Updated crop window for text: x({current_crop_xmin:.0f}-{current_crop_xmax:.0f}), y({current_crop_ymin:.0f}-{current_crop_ymax:.0f})")
                        crop_debug_info['text_bboxes_that_expanded_crop'].append({
                            'uid': bbox_original.get('persistent_uid'),
                            'class': bbox_original.get('class'),
                            'coords_original': (bbox_original['xmin'], bbox_original['ymin'], bbox_original['xmax'], bbox_original['ymax']),
                            'coords_text_box_abs': (text_xmin, text_ymin, text_xmax, text_ymax) # these are absolute to original image
                        })

        # Finalize crop coordinates as integers, ensuring they remain within image boundaries
        crop_abs_xmin = max(0, int(round(current_crop_xmin)))
        crop_abs_ymin = max(0, int(round(current_crop_ymin)))
        crop_abs_xmax = min(original_width, int(round(current_crop_xmax)))
        crop_abs_ymax = min(original_height, int(round(current_crop_ymax)))
        crop_debug_info['final_crop_window_abs'] = (crop_abs_xmin, crop_abs_ymin, crop_abs_xmax, crop_abs_ymax)

        if self.debug:
            print(f"Crop: Original image dims: {original_width}x{original_height}")
            print(f"Crop: Defining bbox for crop (SAM extent): {def_xmin, def_ymin, def_xmax, def_ymax}")
            print(f"Crop: Final calculated absolute crop window (xmin,ymin,xmax,ymax): {crop_abs_xmin}, {crop_abs_ymin}, {crop_abs_xmax}, {crop_abs_ymax}")

        if crop_abs_xmin >= crop_abs_xmax or crop_abs_ymin >= crop_abs_ymax:
            if self.debug:
                print("Crop: Invalid crop region. Returning originals.")
            crop_debug_info['reason_for_no_crop'] = "invalid_region_after_expansion"
            crop_debug_info['cropped_image_dims'] = crop_debug_info['original_image_dims'] # No change
            # Crop was attempted but resulted in an invalid region, still return originals but with debug info
            return image_to_crop, [deepcopy(b) for b in bboxes_to_adjust], crop_debug_info

        cropped_image = image_to_crop[crop_abs_ymin:crop_abs_ymax, crop_abs_xmin:crop_abs_xmax]
        new_height, new_width = cropped_image.shape[:2]
        crop_debug_info['cropped_image_dims'] = (new_width, new_height)
        crop_debug_info['crop_applied'] = True

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
        
        # Ensure final return is the crop_debug_info dictionary
        return cropped_image, adjusted_bboxes, crop_debug_info

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
        connection_points_on_contours = [] # <<< Initialize list to store connection points

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
                        # IMPORTANT: Store the component from processing_bboxes_resized to align coordinate systems
                        # for geometric reasoning with node contours. persistent_uid and semantic_direction
                        # should have been copied by resize_bboxes.
                        target_bbox_to_add = deepcopy(processing_bboxes_resized[i])
                        
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
                            connection_points_on_contours.append(point) # <<< Store the connection point
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
            # Create an empty connection points visualization if no valid nodes
            connection_points_visualization = contour_image_viz.copy() # Start with contour viz
            return [], emptied_mask, enhanced, contour_image_viz, viz_fallback, connection_points_visualization

        max_connections_node_val = max(len(node_data['components']) for node_data in valid_nodes.values())
        
        nodes_with_max = [node_id for node_id, node_data in valid_nodes.items() 
                         if len(node_data['components']) == max_connections_node_val]

        chosen_ground_node_old_id = None
        
        # New Ground Selection Logic:
        # 1. Find all valid nodes connected to a source.
        all_source_connected_candidates = []
        for node_id_candidate, node_data_candidate in valid_nodes.items():
            if any(comp['class'] in self.source_components for comp in node_data_candidate['components']):
                contour_detail = next((c for c in contours if c['id'] == node_id_candidate), None)
                if contour_detail:
                    M = cv2.moments(contour_detail['contour'])
                    if M["m00"] != 0:
                        cY = int(M["m01"] / M["m00"])
                        all_source_connected_candidates.append({'id': node_id_candidate, 'centroid_y': cY})
                    else:
                        all_source_connected_candidates.append({'id': node_id_candidate, 'centroid_y': -float('inf')}) # Should not happen
                else:
                    all_source_connected_candidates.append({'id': node_id_candidate, 'centroid_y': -float('inf')}) # Should not happen

        if all_source_connected_candidates:
            if self.debug:
                print(f"Debug GroundChoice: all_source_connected_candidates BEFORE sort: {all_source_connected_candidates}")
            # 2. From source-connected nodes, pick the one lowest on the screen (largest centroid_y).
            all_source_connected_candidates.sort(key=lambda x: x['centroid_y'], reverse=True)
            if self.debug:
                print(f"Debug GroundChoice: all_source_connected_candidates AFTER sort: {all_source_connected_candidates}")
            chosen_ground_node_old_id = all_source_connected_candidates[0]['id']
            if self.debug:
                chosen_centroid_y_val = all_source_connected_candidates[0]['centroid_y']
                print(f"Debug GroundChoice: CHOSEN old_id_for_ground={chosen_ground_node_old_id} with centroid_y={chosen_centroid_y_val}")
        else:
            # Fallback if no source-connected nodes are found (unlikely for valid circuits)
            # Use the previous fallback: from nodes_with_max, pick the lowest.
            # If nodes_with_max is also problematic, pick the lowest of all valid_nodes.
            if nodes_with_max:
                if len(nodes_with_max) > 1:
                    nodes_with_max_details = []
                    for node_id_candidate in nodes_with_max:
                        contour_detail = next((c for c in contours if c['id'] == node_id_candidate), None)
                        if contour_detail:
                            M = cv2.moments(contour_detail['contour'])
                            if M["m00"] != 0:
                                cY = int(M["m01"] / M["m00"])
                                nodes_with_max_details.append({'id': node_id_candidate, 'centroid_y': cY})
                            else:
                                nodes_with_max_details.append({'id': node_id_candidate, 'centroid_y': -float('inf')})
                        else:
                             nodes_with_max_details.append({'id': node_id_candidate, 'centroid_y': -float('inf')})
                    
                    if nodes_with_max_details:
                        nodes_with_max_details.sort(key=lambda x: x['centroid_y'], reverse=True)
                        chosen_ground_node_old_id = nodes_with_max_details[0]['id']
                    else: 
                        chosen_ground_node_old_id = nodes_with_max[0]
                else: 
                    chosen_ground_node_old_id = nodes_with_max[0]
            elif valid_nodes: # Ultimate fallback: lowest of all valid nodes
                valid_nodes_details = []
                for node_id_candidate in valid_nodes.keys():
                    contour_detail = next((c for c in contours if c['id'] == node_id_candidate), None)
                    if contour_detail:
                        M = cv2.moments(contour_detail['contour'])
                        if M["m00"] != 0:
                            cY = int(M["m01"] / M["m00"])
                            valid_nodes_details.append({'id': node_id_candidate, 'centroid_y': cY})
                        else:
                            valid_nodes_details.append({'id': node_id_candidate, 'centroid_y': -float('inf')})
                    else:
                        valid_nodes_details.append({'id': node_id_candidate, 'centroid_y': -float('inf')})
                if valid_nodes_details:
                    valid_nodes_details.sort(key=lambda x: x['centroid_y'], reverse=True)
                    chosen_ground_node_old_id = valid_nodes_details[0]['id']
                else:
                     chosen_ground_node_old_id = list(valid_nodes.keys())[0] if valid_nodes else None
            # Ensure chosen_ground_node_old_id is not None if valid_nodes exist
            if chosen_ground_node_old_id is None and valid_nodes:
                chosen_ground_node_old_id = list(valid_nodes.keys())[0] # Default to first valid node

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

        # Create visualization for connection points
        connection_points_visualization = contour_image_viz.copy() # Start with the contour viz image
        for p_conn in connection_points_on_contours:
            cv2.circle(connection_points_visualization, p_conn, radius=5, color=(0, 255, 255), thickness=-1) # Cyan circles

        return new_nodes_list, emptied_mask, enhanced, contour_image_viz, final_node_viz_image, connection_points_visualization

    def generate_netlist_from_nodes(self, node_list):
        netlist = []
        id_count = 1
        component_counters = {component_type: 1 for component_type in set(self.netlist_map.values()) if component_type}
        processed_components = set()
        
        # Create a quick lookup for node centroids
        # The contours are in the same space as the component bboxes stored in node_list[i]['components']
        # due to changes in get_node_connections.
        node_centroids = {}
        for node_data in node_list:
            node_id = node_data['id']
            contour = node_data.get('contour')
            if contour is not None and len(contour) > 0:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    node_centroids[node_id] = (cx, cy)
                else:
                    # Fallback if moment is zero, though unlikely for valid contours
                    # Use the first point of the contour or a default
                    node_centroids[node_id] = tuple(contour[0][0]) if len(contour[0]) > 0 else (0,0)
            else:
                # Fallback if no contour, though all valid nodes should have one
                node_centroids[node_id] = None 
                if self.debug:
                    print(f"Warning: Node {node_id} has no contour for centroid calculation.")

        # Removed old first pass for source_positive_node
        
        # Process components
        for n_idx, n_data in enumerate(node_list):
            node_components = n_data['components']
            current_node_id = n_data['id']
            
            for component in node_components:
                component_class = component.get('class')
                # Bbox for this component is already in the same space as node contours
                component_bbox = {
                    'xmin': component['xmin'], 'ymin': component['ymin'],
                    'xmax': component['xmax'], 'ymax': component['ymax']
                }
                component_semantic_direction = component.get('semantic_direction', 'UNKNOWN')
                persistent_uid = component.get('persistent_uid')
                semantic_reason = component.get('semantic_reason', 'UNKNOWN') # Get the reason

                if not persistent_uid: # Should always have this
                    if self.debug: print(f"Warning: Component {component_class} missing persistent_uid. Skipping.")
                    continue

                # Check if component should be skipped
                is_ignorable_class = component_class in ['text', 'explanatory', 'junction', 'crossover', 'terminal']
                is_already_processed = persistent_uid in processed_components
                if is_ignorable_class or is_already_processed:
                    continue
                    
                processed_components.add(persistent_uid)
                # Determine component_type_prefix considering semantic_reason
                yolo_class = component_class # Original class from YOLO
                reason = component.get('semantic_reason', 'UNKNOWN')
                prospective_prefix = self.netlist_map.get(yolo_class, 'UN') # Default prefix

                if yolo_class in self.voltage_classes_names and reason == "ARROW":
                    # If YOLO said voltage but LLaMA saw an arrow, treat as current source for prefix
                    # Assuming independent current source for now, map to 'I'
                    # Dependent sources might have other symbols (diamond) which LLaMA isn't specifically looking for yet
                    prospective_prefix = 'I' 
                    if self.debug:
                        print(f"InitialNetlist: YOLO class '{yolo_class}' with reason ARROW. Using prefix 'I'. UID: {persistent_uid}")
                elif yolo_class in self.current_source_classes_names and reason == "SIGN":
                    # If YOLO said current but LLaMA saw a sign, treat as voltage source for prefix
                    prospective_prefix = 'V'
                    if self.debug:
                        print(f"InitialNetlist: YOLO class '{yolo_class}' with reason SIGN. Using prefix 'V'. UID: {persistent_uid}")

                component_type_prefix = prospective_prefix
                
                if not component_type_prefix: # Should be caught by default 'UN' earlier if mapping failed
                    if self.debug:
                        print(f"Warning: Component {yolo_class} resulted in empty component_type_prefix. Skipping UID {persistent_uid}.")
                    continue
                    
                # Find the other node this component is connected to
                other_node_id = None
                for other_n_data in node_list:
                    if other_n_data['id'] != current_node_id:
                        # Check against persistent_uid as components are deepcopied
                        if any(c.get('persistent_uid') == persistent_uid for c in other_n_data['components']):
                            other_node_id = other_n_data['id']
                            break
                            
                if other_node_id is None:
                    if self.debug:
                        print(f"Warning: Could not find second node for component UID {persistent_uid} ({component_class}). Skipping.")
                    continue
                
                # Get centroids for the connected nodes
                current_node_centroid = node_centroids.get(current_node_id)
                other_node_centroid = node_centroids.get(other_node_id)

                if current_node_centroid is None or other_node_centroid is None:
                    if self.debug:
                        print(f"Warning: Missing centroid for one of the nodes ({current_node_id}, {other_node_id}) connected to UID {persistent_uid}. Using arbitrary node order.")
                    # Fallback: use original IDs, no directed assignment possible
                    assigned_node1_id, assigned_node2_id = current_node_id, other_node_id
                else:
                    # Determine primary and secondary nodes using semantic direction
                    # The component_bbox is already in the same coordinate space as centroids.
                    primary_centroid, secondary_centroid = self._get_terminal_nodes_relative_to_bbox(
                        component, # Pass the full component dict
                        component_semantic_direction, 
                        current_node_centroid, 
                        other_node_centroid, 
                        component_class,
                        semantic_reason # Pass the reason
                    )
                    
                    # Map back from centroids to original node IDs
                    if primary_centroid == current_node_centroid:
                        assigned_node1_id, assigned_node2_id = current_node_id, other_node_id
                    else:
                        assigned_node1_id, assigned_node2_id = other_node_id, current_node_id

                # Grounding and final node assignment
                # If it's a ground component, one node must be 0.
                if component_class in ['gnd', 'vss']:
                    # The non-ground node is assigned_node1_id (or assigned_node2_id if it was the primary one)
                    # The other one becomes 0.
                    if assigned_node1_id == 0: # Should not happen if 0 is not a normal node ID
                         true_node = assigned_node2_id
                    else:
                         true_node = assigned_node1_id # Assume this is the actual connection point
                    node_1, node_2 = true_node, 0
                # For other components, directly use the assigned primary and secondary nodes
                # The directionality is already embedded in assigned_node1_id (N+) and assigned_node2_id (N-)
                else:
                    node_1 = assigned_node1_id
                    node_2 = assigned_node2_id
                
                # For AC sources, value might be complex (e.g. "10V:30deg"), handle later by Gemini.
                # For DC sources, value could be simple voltage.
                value = "None"  # Placeholder value, to be filled by fix_netlist using Gemini
                
                # Get unique component ID (R1, C1, etc.)
                if component_type_prefix not in component_counters:
                     component_counters[component_type_prefix] = 1 # Should be pre-initialized, but as a safeguard
                comp_num = component_counters[component_type_prefix]
                component_counters[component_type_prefix] += 1
                # component_full_id = f"{component_type_prefix}{comp_num}" # This is just for logging/debug if needed
                
                line = {
                    'component_type': component_type_prefix,
                    'component_num': comp_num,
                    'node_1': node_1,
                    'node_2': node_2,
                    'value': value
                }
                # Add all original component fields (class, semantic_direction, persistent_uid, original bbox, etc.)
                # from the component dict that was processed (which is in contour space).
                # The original `component` dict already has these.
                line.update(deepcopy(component)) 
                # Ensure xmin, ymin etc. are not overwritten if present in component by line.update(component)
                # by ensuring component_bbox fields are not in `line` before update, or handle carefully.
                # For now, component dict has xmin,ymin (in contour space) and other important fields.
                
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

    def _encode_image_for_llama(self, img_array_rgb):
        """Convert numpy array (RGB) to base64 encoded image for LLaMA."""
        img_pil = PILImage.fromarray(img_array_rgb)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')

    def _get_terminal_nodes_relative_to_bbox(self, component_data, # Changed from component_bbox
                                             semantic_direction_arg, 
                                             node1_centroid, node2_centroid, component_class_name_arg,
                                             semantic_reason_arg="UNKNOWN"): 
        """
        Determines which node centroid corresponds to the primary terminal of a component 
        based on its semantic direction and bounding box.

        Args:
            component_data (dict): The full component dictionary, including coordinates, class, UID, etc.
            semantic_direction_arg (str): "UP", "DOWN", "LEFT", "RIGHT", or "UNKNOWN".
            node1_centroid (tuple): (x, y) coordinates of the first connected node's centroid.
            node2_centroid (tuple): (x, y) coordinates of the second connected node's centroid.
            component_class_name_arg (str): The class name (e.g., 'voltage.dc', 'diode') from the component_data.
            semantic_reason_arg (str): "SIGN", "ARROW", or "UNKNOWN". Default is "UNKNOWN".

        Returns:
            tuple: (primary_node_centroid, secondary_node_centroid) if direction is clear,
                   else (node1_centroid, node2_centroid) as a fallback (e.g., for UNKNOWN direction).
                   For voltage/current sources: primary is positive/source-from.
                   For diodes: primary is anode.
        """
        if not node1_centroid or not node2_centroid:
            # Should not happen if nodes are valid
            return node1_centroid, node2_centroid 

        # Extract needed info from component_data
        component_class_name = component_data.get('class', component_class_name_arg) 
        persistent_uid_debug = component_data.get('persistent_uid')

        is_diode = component_class_name in self.diode_classes_names
        is_yolo_voltage_source = component_class_name in self.voltage_classes_names
        is_yolo_current_source = component_class_name in self.current_source_classes_names 

        acts_like_arrow = is_yolo_current_source or (is_yolo_voltage_source and semantic_reason_arg == "ARROW")
        acts_like_sign_voltage = is_yolo_voltage_source and semantic_reason_arg != "ARROW"

        if self.debug:
            print(f"NodeOrderDebug for {component_class_name}, UID: {persistent_uid_debug}:")
            print(f"  semantic_direction_arg: {semantic_direction_arg}, semantic_reason_arg: {semantic_reason_arg}")
            print(f"  is_yolo_voltage_source: {is_yolo_voltage_source}, is_yolo_current_source: {is_yolo_current_source}, is_diode: {is_diode}")
            print(f"  acts_like_arrow: {acts_like_arrow}, acts_like_sign_voltage: {acts_like_sign_voltage}")
            print(f"  Condition for default: {semantic_direction_arg == 'UNKNOWN' or not (acts_like_arrow or acts_like_sign_voltage or is_diode)}")

        primary_node_candidate1 = node1_centroid # n1_in
        secondary_node_candidate1 = node2_centroid # n2_in

        if semantic_direction_arg == "UNKNOWN" or not (acts_like_arrow or acts_like_sign_voltage or is_diode):
            if self.debug and semantic_direction_arg != "UNKNOWN":
                 print(f"Warning: Node ordering for {component_class_name} (reason: {semantic_reason_arg}) with direction {semantic_direction_arg} not explicitly handled. Using default node order.")
            return node1_centroid, node2_centroid

        n1x, n1y = node1_centroid
        n2x, n2y = node2_centroid

        swapped = False
        
        # Log inputs to decision logic for the specific component if its UID matches
        if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
            print(f"UID_MATCH_DEBUG ({persistent_uid_debug}): n1_in={node1_centroid}, n2_in={node2_centroid}, dir={semantic_direction_arg}")

        if semantic_direction_arg == "UP": 
            comparison_result = n1y < n2y
            if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
                print(f"UID_MATCH_DEBUG (UP): n1y({n1y}) < n2y({n2y}) is {comparison_result}")
            if comparison_result: 
                swapped = True
        elif semantic_direction_arg == "DOWN": 
            comparison_result = n1y > n2y
            if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
                print(f"UID_MATCH_DEBUG (DOWN): n1y({n1y}) > n2y({n2y}) is {comparison_result}")
            if comparison_result: 
                swapped = True
        elif semantic_direction_arg == "LEFT": 
            comparison_result = n1x < n2x
            if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
                print(f"UID_MATCH_DEBUG (LEFT): n1x({n1x}) < n2x({n2x}) is {comparison_result}")
            if comparison_result: 
                swapped = True
        elif semantic_direction_arg == "RIGHT": 
            comparison_result = n1x > n2x
            if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
                print(f"UID_MATCH_DEBUG (RIGHT): n1x({n1x}) > n2x({n2x}) is {comparison_result}")
            if comparison_result: 
                swapped = True
        else: 
            if self.debug:
                print(f"Debug: Unhandled semantic_direction '{semantic_direction_arg}' in node ordering. Defaulting.")
            return node1_centroid, node2_centroid

        if self.debug and persistent_uid_debug == 'voltage.ac_919_239_1025_341':
            print(f"UID_MATCH_DEBUG ({persistent_uid_debug}): final swapped={swapped}")

        if swapped:
            return secondary_node_candidate1, primary_node_candidate1 # node2 becomes primary
        else:
            return primary_node_candidate1, secondary_node_candidate1 # node1 remains primary

    def _get_semantic_direction_from_llama(self, component_crop_rgb, component_class_name):
        """
        Analyzes a component crop with LLaMA via Groq to determine its direction.
        'component_class_name' is the string name like 'voltage.dc'.
        """
        if not self.groq_client:
            if self.debug:
                print("Groq client not available. Skipping LLaMA direction analysis.")
            return "UNKNOWN", "UNKNOWN"

        base64_image = self._encode_image_for_llama(component_crop_rgb)
        prompt = None
        model_to_use = "meta-llama/llama-4-scout-17b-16e-instruct" 

        if component_class_name in self.voltage_classes_names:
            prompt = """Analyze this image.

Focus on identifying the following key elements:
1. The + (plus) and - (minus) symbols or arrow if present
2. Their relative positions in the image (top, bottom, left, right)

Return a JSON object with these fields:
- symbol_positions: Describe the exact locations of + and - symbols. If there's an arrow instead, write "ARROW"
- direction: ONE of [UP, DOWN, LEFT, RIGHT] determined by these rules:
  * For +/- symbols:
    - If + is at bottom → direction: "UP"
    - If + is at top → direction: "DOWN"
    - If + is at left → direction: "RIGHT"
    - If + is at right → direction: "LEFT"
  * For voltage arrow:
    - Arrow pointing up → direction: "UP"
    - Arrow pointing down → direction: "DOWN"
    - Arrow pointing left → direction: "LEFT"
    - Arrow pointing right → direction: "RIGHT"
- reason: ONE of ["SIGN", "ARROW"] indicating if direction was based on +/- symbols or an arrow.

Example responses:
{"symbol_positions": "+ at bottom, - at top", "direction": "UP", "reason": "SIGN"}
{"symbol_positions": "ARROW", "direction": "RIGHT", "reason": "ARROW"}
"""
        elif component_class_name in self.diode_classes_names:
            prompt = """Analyze this image.

A diode symbol consists of:
1. A triangle (▶) pointing in the direction of current flow
2. A bar (|) perpendicular to the direction of flow

Focus on identifying:
1. The orientation of the triangle-bar symbol
2. The direction the triangle is pointing (this is the direction of current flow)

Return a JSON object with ONE field:
- direction: ONE of [UP, DOWN, LEFT, RIGHT] based on where the triangle points:
  * Triangle points up → direction: "UP"
  * Triangle points down → direction: "DOWN"
  * Triangle points left → direction: "LEFT"
  * Triangle points right → direction: "RIGHT"

Example responses:
{"direction": "RIGHT"}  // For triangle pointing right →
{"direction": "UP"}     // For triangle pointing up ↑
"""
        else:
            if self.debug:
                print(f"No specific LLaMA prompt for component class: {component_class_name}. Skipping LLaMA.")
            return "UNKNOWN", "UNKNOWN" # Not a component type we analyze for direction with LLaMA with these prompts

        try:
            if self.debug:
                print(f"LLaMA_QUERY_V3 ({model_to_use}) for direction of {component_class_name}...")
            
            llm_start_time = time.time()
            chat_completion = self.groq_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0, 
                max_tokens=1024, 
                top_p=0.95,         
                stream=False,
                response_format={"type": "json_object"} 
            )
            llm_time = time.time() - llm_start_time
            response_content = chat_completion.choices[0].message.content
            
            if self.debug:
                print(f"LLaMA_RESPONSE_V3 for {component_class_name} (took {llm_time:.2f}s): {response_content}")
            
            parsed_response = json.loads(response_content)
            # The direction field seems to be directly the value like "UP", "DOWN" etc.
            # Also, the prompt for voltage sources might return 'symbol_positions', 
            # but we are primarily interested in the 'direction' field for the netlist.
            # We will prioritize the 'direction' field if present.
            direction = parsed_response.get("direction")
            reason = parsed_response.get("reason") # Get the reason

            if direction: # Only return reason if direction is valid
                 return str(direction).upper(), str(reason).upper() if reason else "UNKNOWN" # Return both
            else:
                if self.debug:
                    print(f"LLaMA_RESPONSE_V3_NO_DIRECTION for {component_class_name}. Full response: {parsed_response}")
                return "UNKNOWN", "UNKNOWN" # Fallback on error for both
            
        except Exception as e:
            if self.debug: # Added this check for consistency
                print(f"LLaMA_ERROR_V3 for {component_class_name} direction: {e}")
            return "UNKNOWN", "UNKNOWN" # Fallback on error for both

    def _enrich_bboxes_with_directions(self, image_rgb, bboxes):
        """
        Iterates through bboxes, and for relevant components (sources, diodes based on classes_of_interest),
        uses LLaMA to determine semantic direction and adds it to the bbox dict.
        Modifies bboxes in-place.
        Assumes image_rgb is the original, full image in RGB format.
        Uses numeric class IDs and confidence threshold from code.
        """
        if not self.groq_client:
            if self.debug:
                print("Groq client not initialized. Skipping semantic direction enrichment.")
            return
        

        numeric_classes_of_interest = {6, 7, 8, 17, 18, 19, 29, 22} 
        confidence_threshold = 0.4 

        for bbox in bboxes:
            yolo_numeric_cls_id = bbox.get('_yolo_class_id_temp') # Assuming this was added during initial YOLO parsing
            if yolo_numeric_cls_id is None:
                 # If not present, try to find it by matching string class name back to ID
                 # This is less ideal but a fallback.
                string_class_name_for_id_lookup = bbox.get('class')
                for num_id, name_str in self.yolo_class_names_map.items():
                    if name_str == string_class_name_for_id_lookup:
                        yolo_numeric_cls_id = num_id
                        break
            
            confidence = bbox.get('confidence', 0.0)
            component_string_class_name = bbox.get('class') # String name like 'voltage.dc'

            if yolo_numeric_cls_id is not None and \
               yolo_numeric_cls_id in numeric_classes_of_interest and \
               confidence >= confidence_threshold and \
               component_string_class_name is not None and \
               component_string_class_name in self.classes_of_interest_names:
                
                # Crop the component from the original image
                orig_xmin, orig_ymin = int(bbox['xmin']), int(bbox['ymin'])
                orig_xmax, orig_ymax = int(bbox['xmax']), int(bbox['ymax'])
                
                # Define padding for the LLaMA crop
                llama_crop_padding = 15  # Pixels
                h, w = image_rgb.shape[:2]

                # Apply padding and ensure coordinates are within image bounds
                crop_xmin = max(0, orig_xmin - llama_crop_padding)
                crop_ymin = max(0, orig_ymin - llama_crop_padding)
                crop_xmax = min(w, orig_xmax + llama_crop_padding)
                crop_ymax = min(h, orig_ymax + llama_crop_padding)

                # Ensure crop coordinates are valid after padding
                if crop_xmin >= crop_xmax or crop_ymin >= crop_ymax:
                    if self.debug:
                        print(f"Skipping LLaMA for {component_string_class_name} due to invalid crop dimensions: {bbox}")
                    bbox['semantic_direction'] = 'UNKNOWN'
                    bbox['semantic_reason'] = 'UNKNOWN'
                    continue

                component_crop_rgb = image_rgb[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                
                if self.debug and bbox.get('persistent_uid'):
                    # Store a copy of the image that will be/was sent to LLaMA
                    self.last_llama_input_images[bbox['persistent_uid']] = component_crop_rgb.copy()

                if component_crop_rgb.size == 0:
                    if self.debug:
                        print(f"Skipping LLaMA for {component_string_class_name} due to empty crop: {bbox.get('persistent_uid')}")
                    bbox['semantic_direction'] = 'UNKNOWN' # Explicitly string UNKNOWN
                    bbox['semantic_reason'] = 'UNKNOWN'
                    continue

                # Get semantic direction and reason from LLaMA
                direction_from_llama, reason_from_llama = self._get_semantic_direction_from_llama(component_crop_rgb, component_string_class_name)
                
                if self.debug:
                    print(f"EnrichDebug UID {bbox.get('persistent_uid')}: LLaMA returned dir='{direction_from_llama}', reason='{reason_from_llama}'")

                bbox['semantic_direction'] = direction_from_llama # Directly assign what LLaMA helper returned
                bbox['semantic_reason'] = reason_from_llama   # Directly assign
                
                if self.debug:
                    print(f"EnrichDebug UID {bbox.get('persistent_uid')}: Stored in bbox dir='{bbox['semantic_direction']}', reason='{bbox['semantic_reason']}'")
            else:
                # For components not analyzed for direction, set a default or skip
                bbox['semantic_direction'] = None
                bbox['semantic_reason'] = None # Ensure reason is also None
        # No explicit return, bboxes list is modified in-place.