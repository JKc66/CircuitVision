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
from PIL import Image

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

# Load environment variables
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
        self.non_components = set(['text', 'junction', 'crossover', 'vss', 'explanatory', 'circuit'])
        self.source_components = set(['voltage.ac', 'voltage.dc', 'voltage.dependent', 'current.dc', 'current.dependent'])
        
        # Add property to store last SAM2 output
        self.last_sam2_output = None
        
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
        self.yolo_class_names_map = self.yolo.model.names # Get mapping from class ID to name from YOLO model
        # Define classes of interest by string name for LLaMA processing
        self.llama_classes_of_interest_names = {
            'voltage.dc', 'voltage.ac', 
            'diode', 'diode.light_emitting', 'diode.zener',
            'transistor.bjt', 
            'unknown' 
        }
        # Convert string names to numeric IDs for efficient checking in _enrich_bboxes_with_directions
        self.llama_numeric_classes_of_interest = set()
        for num_id, name_str in self.yolo_class_names_map.items():
            if name_str in self.llama_classes_of_interest_names:
                self.llama_numeric_classes_of_interest.add(num_id)
        
        if self.debug:
            print(f"LLaMA numeric classes of interest for direction: {self.llama_numeric_classes_of_interest}")

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
        image_for_enumeration = image.copy() # Work on a copy
        if excluded_labels is None:
            excluded_labels = self.non_components

        # Get the image dimensions
        image_height, image_width = image_for_enumeration.shape[:2]
        # Convert image to grayscale for background intensity checking - IF it's not already grayscale
        if len(image_for_enumeration.shape) == 3 and image_for_enumeration.shape[2] == 3:
            gray_image = cv2.cvtColor(image_for_enumeration, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_for_enumeration.copy() # Assume it's already grayscale if not 3-channel


        # Calculate a relative font scale based on the image height
        font_scale = max(0.4, image_height / 900.0) # Further reduced divisor for smaller font, min lowered
        thickness = int(max(1, image_height / 600.0)) # Further reduced divisor for thinner font

        bbox_counter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color_cv = (0, 0, 255)  # Red color for the text in BGR for OpenCV

        # Helper: Calculate overlap area between two rectangles (xmin, ymin, xmax, ymax)
        def calculate_overlap_area(box1_coords, box2_coords):
            x1_min, y1_min, x1_max, y1_max = box1_coords
            x2_min, y2_min, x2_max, y2_max = box2_coords

            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)

            inter_width = max(0, inter_xmax - inter_xmin)
            inter_height = max(0, inter_ymax - inter_ymin)
            return inter_width * inter_height

        # Helper: Calculate background suitability
        def calculate_background_score_for_text(x_coord, y_coord, text_w, text_h):
            # Ensure the coordinates for region extraction are valid integers
            y_start, y_end = int(round(y_coord)), int(round(y_coord + text_h))
            x_start, x_end = int(round(x_coord)), int(round(x_coord + text_w))

            if (y_start < 0 or x_start < 0 or
                y_end > image_height or
                x_end > image_width):
                return float('-inf') # Heavily penalize out-of-bounds

            region = gray_image[y_start:y_end, x_start:x_end]
            
            if region.size == 0: # Should not happen if in bounds and text_w/h > 0
                return float('-inf')
            
            # New logic: Simply return a constant positive score if the region is valid for placement.
            # This means all valid regions will be treated equally by the sorting key that used to use this score.
            # The first geometrically valid position found will effectively be chosen.
            return 1.0 # Constant positive score for any valid patch

        # Helper function to check if two bounding boxes are "close" or overlapping
        def are_bboxes_proximal(bbox1_dict_or_tuple, bbox2_dict_or_tuple, proximity_threshold=30):
            if isinstance(bbox1_dict_or_tuple, dict):
                xmin1, ymin1, xmax1, ymax1 = bbox1_dict_or_tuple['xmin'], bbox1_dict_or_tuple['ymin'], bbox1_dict_or_tuple['xmax'], bbox1_dict_or_tuple['ymax']
            else: # tuple
                xmin1, ymin1, xmax1, ymax1 = bbox1_dict_or_tuple

            if isinstance(bbox2_dict_or_tuple, dict):
                xmin2, ymin2, xmax2, ymax2 = bbox2_dict_or_tuple['xmin'], bbox2_dict_or_tuple['ymin'], bbox2_dict_or_tuple['xmax'], bbox2_dict_or_tuple['ymax']
            else: # tuple
                xmin2, ymin2, xmax2, ymax2 = bbox2_dict_or_tuple

            # Check for direct overlap first
            if not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2):
                return True # They overlap

            # Check proximity (distance between closest edges)
            # Horizontal distance
            if xmax1 < xmin2: # bbox1 is to the left of bbox2
                h_dist = xmin2 - xmax1
            elif xmin1 > xmax2: # bbox1 is to the right of bbox2
                h_dist = xmin1 - xmax2
            else: # Overlap or aligned vertically
                h_dist = 0
            
            # Vertical distance
            if ymax1 < ymin2: # bbox1 is above bbox2
                v_dist = ymin2 - ymax1
            elif ymin1 > ymax2: # bbox1 is below bbox2
                v_dist = ymin1 - ymax2
            else: # Overlap or aligned horizontally
                v_dist = 0

            # If one dimension overlaps, we only care about the distance in the other dimension
            if h_dist == 0: # Aligned or overlapping horizontally
                return v_dist <= proximity_threshold
            if v_dist == 0: # Aligned or overlapping vertically
                return h_dist <= proximity_threshold
            
            # If separated in both dimensions, check diagonal distance (approximate)
            # More accurate would be distance between closest corners/edges, but this is simpler
            # and direct edge distance check is more robust for rectangular proximity.
            # For now, if direct edge distances are too large, consider them not proximal enough.
            return h_dist <= proximity_threshold and v_dist <= proximity_threshold

        # New find_optimal_position logic
        def find_optimal_position(
            component_to_label_bbox_dict,
            all_other_comp_bboxes_list_dicts,
            static_text_elements_schematic_bboxes_tuples, # list of (xmin,ymin,xmax,ymax)
            already_drawn_numbers_bboxes_tuples, # list of (xmin,ymin,xmax,ymax)
            txt_w, txt_h
        ):
            comp_xmin = component_to_label_bbox_dict['xmin']
            comp_ymin = component_to_label_bbox_dict['ymin']
            comp_xmax = component_to_label_bbox_dict['xmax']
            comp_ymax = component_to_label_bbox_dict['ymax']
            comp_rect_for_check = (comp_xmin, comp_ymin, comp_xmax, comp_ymax)

            comp_xc = comp_xmin + (comp_xmax - comp_xmin) // 2
            comp_yc = comp_ymin + (comp_ymax - comp_ymin) // 2
            
            text_half_w, text_half_h = txt_w // 2, txt_h // 2
            placement_padding = 5 # Reverted to 5 for a bit more space

            candidate_positions_with_names = {
                # Order can influence ties if distances are identical, but primary sort is distance
                "right_middle": (comp_xmax + placement_padding, comp_yc - text_half_h),
                "left_middle": (comp_xmin - txt_w - placement_padding, comp_yc - text_half_h),
                "top_center": (comp_xc - text_half_w, comp_ymin - txt_h - placement_padding),
                "bottom_center": (comp_xc - text_half_w, comp_ymax + placement_padding),
                "top_right_corner_out": (comp_xmax + placement_padding, comp_ymin - txt_h), 
                "top_left_corner_out": (comp_xmin - txt_w - placement_padding, comp_ymin - txt_h),
                "bottom_right_corner_out": (comp_xmax + placement_padding, comp_ymax),
                "bottom_left_corner_out": (comp_xmin - txt_w - placement_padding, comp_ymax),
            }
            
            permissible_positions = []

            for pos_name, (pos_x, pos_y) in candidate_positions_with_names.items():
                pos_x_int, pos_y_int = int(round(pos_x)), int(round(pos_y))
                current_text_rect = (pos_x_int, pos_y_int, pos_x_int + txt_w, pos_y_int + txt_h)

                if not (0 <= pos_x_int < image_width - txt_w and 0 <= pos_y_int < image_height - txt_h):
                    if self.debug: print(f"Pos {pos_name} invalid: Out of bounds for UID {component_to_label_bbox_dict.get('persistent_uid')}")
                    continue

                if calculate_overlap_area(current_text_rect, comp_rect_for_check) > 0:
                    if self.debug: print(f"Pos {pos_name} invalid: Overlaps its own component UID {component_to_label_bbox_dict.get('persistent_uid')}")
                    continue
                
                overlaps_other_component = False
                for other_comp_bbox_dict in all_other_comp_bboxes_list_dicts:
                    other_comp_rect = (other_comp_bbox_dict['xmin'], other_comp_bbox_dict['ymin'], other_comp_bbox_dict['xmax'], other_comp_bbox_dict['ymax'])
                    if calculate_overlap_area(current_text_rect, other_comp_rect) > 0:
                        if self.debug: print(f"Pos {pos_name} invalid: Overlaps OTHER component UID {other_comp_bbox_dict.get('persistent_uid')} for labeling UID {component_to_label_bbox_dict.get('persistent_uid')}")
                        overlaps_other_component = True
                        break
                if overlaps_other_component:
                    continue

                overlaps_static_text = False
                for static_text_rect in static_text_elements_schematic_bboxes_tuples:
                    if calculate_overlap_area(current_text_rect, static_text_rect) > 0:
                        if self.debug: print(f"Pos {pos_name} invalid: Overlaps static text element at {static_text_rect} for labeling UID {component_to_label_bbox_dict.get('persistent_uid')}")
                        overlaps_static_text = True
                        break
                if overlaps_static_text:
                    continue

                overlaps_drawn_number = False
                for drawn_num_rect in already_drawn_numbers_bboxes_tuples:
                    if calculate_overlap_area(current_text_rect, drawn_num_rect) > 0:
                        if self.debug: print(f"Pos {pos_name} invalid: Overlaps already drawn number at {drawn_num_rect} for labeling UID {component_to_label_bbox_dict.get('persistent_uid')}")
                        overlaps_drawn_number = True
                        break
                if overlaps_drawn_number:
                    continue
                
                # bg_score is now just a check for validity (1.0) vs invalidity (-inf)
                bg_score_check = calculate_background_score_for_text(pos_x_int, pos_y_int, txt_w, txt_h)
                if bg_score_check > float('-inf'):
                    # Calculate distance from component center to text center
                    text_center_x = pos_x_int + txt_w // 2
                    text_center_y = pos_y_int + txt_h // 2
                    distance = np.sqrt((comp_xc - text_center_x)**2 + (comp_yc - text_center_y)**2)
                    permissible_positions.append({
                        'x': pos_x_int, 
                        'y': pos_y_int, 
                        'name': pos_name,
                        'distance': distance # Store distance for sorting
                    })
                elif self.debug:
                     print(f"Pos {pos_name} for UID {component_to_label_bbox_dict.get('persistent_uid')} had bad bg_score/invalid patch, rect: {current_text_rect}")

            if not permissible_positions:
                if self.debug: print(f"No permissible adjacent external position found for component UID {component_to_label_bbox_dict.get('persistent_uid')}")
                return None 

            # Sort permissible positions by distance (ascending - closest first)
            permissible_positions.sort(key=lambda p: p['distance'])
            
            best_pos = permissible_positions[0] # Choose the closest one
            if self.debug: 
                print(f"Best position for UID {component_to_label_bbox_dict.get('persistent_uid')} is {best_pos['name']} ({best_pos['x']},{best_pos['y']}) with distance {best_pos['distance']:.2f}")
            return best_pos['x'], best_pos['y']

        # Prepare lists for find_optimal_position
        all_input_bboxes_dicts = []
        if bboxes is None: 
            all_input_bboxes_dicts = self.bboxes(image_for_enumeration) 
            all_input_bboxes_dicts = non_max_suppression_by_area(all_input_bboxes_dicts, iou_threshold=0.6)
        else:
            all_input_bboxes_dicts = deepcopy(bboxes) 

        static_text_elements_bboxes_tuples = [] # Store as tuples (xmin,ymin,xmax,ymax)
        component_bboxes_for_enumeration_dicts = [] # Store as list of dicts

        for bbox_item_dict in all_input_bboxes_dicts:
            if excluded_labels and bbox_item_dict['class'] in excluded_labels:
                static_text_elements_bboxes_tuples.append(
                    (bbox_item_dict['xmin'], bbox_item_dict['ymin'], bbox_item_dict['xmax'], bbox_item_dict['ymax'])
                )
            else:
                component_bboxes_for_enumeration_dicts.append(bbox_item_dict)
        
        output_bbox_ids_with_visual_enum = []
        drawn_numbers_actual_bboxes_tuples = [] 

        for current_component_bbox_dict in component_bboxes_for_enumeration_dicts:
            current_comp_uid_debug = current_component_bbox_dict.get('persistent_uid', "UID_UNKNOWN")

            bbox_counter += 1
            text_to_draw = f"{bbox_counter}"
            (current_text_width, current_text_height), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
            
            other_component_bboxes_for_check_dicts = [
                b_dict for b_dict in component_bboxes_for_enumeration_dicts
                if b_dict.get('persistent_uid', f"{b_dict['class']}_{b_dict['xmin']}_{b_dict['ymin']}") != current_component_bbox_dict.get('persistent_uid', f"{current_component_bbox_dict['class']}_{current_component_bbox_dict['xmin']}_{current_component_bbox_dict['ymin']}")
            ]
            
            optimal_pos_xy_tuple = find_optimal_position(
                current_component_bbox_dict,
                other_component_bboxes_for_check_dicts,
                static_text_elements_bboxes_tuples,
                drawn_numbers_actual_bboxes_tuples,
                current_text_width, current_text_height
            )
            
            final_pos_x, final_pos_y = -1, -1

            if optimal_pos_xy_tuple:
                final_pos_x, final_pos_y = optimal_pos_xy_tuple
            else:
                # Fallback: Place slightly above and to the right of the component's top-left corner
                comp_xmin_val = current_component_bbox_dict['xmin']
                comp_ymin_val = current_component_bbox_dict['ymin']
                
                fallback_x = comp_xmin_val + 3 # Small offset to the right
                fallback_y = comp_ymin_val - current_text_height - 3 # Small offset above
                
                final_pos_x = max(0, min(fallback_x, image_width - current_text_width))
                final_pos_y = max(0, min(fallback_y, image_height - current_text_height))

                # Ensure fallback doesn't overlap with its own component if possible (simple check)
                fallback_text_rect = (final_pos_x, final_pos_y, final_pos_x + current_text_width, final_pos_y + current_text_height)
                comp_rect = (current_component_bbox_dict['xmin'], current_component_bbox_dict['ymin'], current_component_bbox_dict['xmax'], current_component_bbox_dict['ymax'])
                if calculate_overlap_area(fallback_text_rect, comp_rect) > 0:
                    # If simple fallback overlaps, try just top-left of image as last resort
                    final_pos_x = 5 
                    final_pos_y = 5 + (bbox_counter -1) * (current_text_height + 2) # Cascade if many fallbacks
                    final_pos_x = max(0, min(final_pos_x, image_width - current_text_width))
                    final_pos_y = max(0, min(final_pos_y, image_height - current_text_height))


                if self.debug:
                    print(f"UID {current_comp_uid_debug}: Using fallback position ({final_pos_x},{final_pos_y})")

            draw_y_coord = final_pos_y + current_text_height
            
            cv2.putText(image_for_enumeration, text_to_draw, (final_pos_x, draw_y_coord), font,
                       font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
            cv2.putText(image_for_enumeration, text_to_draw, (final_pos_x, draw_y_coord), font,
                       font_scale, text_color_cv, thickness, cv2.LINE_AA)
            
            drawn_numbers_actual_bboxes_tuples.append((final_pos_x, final_pos_y, final_pos_x + current_text_width, final_pos_y + current_text_height))
            
            component_with_visual_id = deepcopy(current_component_bbox_dict)
            component_with_visual_id['id'] = bbox_counter 
            output_bbox_ids_with_visual_enum.append(component_with_visual_id)
            
        if self.debug:
            debug_display_img = image_for_enumeration
            if len(image_for_enumeration.shape) == 3 and image_for_enumeration.shape[2] == 3:
                 debug_display_img = cv2.cvtColor(image_for_enumeration, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 12)) 
            plt.imshow(debug_display_img)
            plt.title(f"Enumerated Components (New Logic) - {len(output_bbox_ids_with_visual_enum)} items numbered")
            plt.axis("off")
            plt.show()
            
        return image_for_enumeration, output_bbox_ids_with_visual_enum

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
    
        
        image_copy = mask.copy()  # Work with a copy of the potentially inverted mask
        
        for bbox in bboxes:
            # If the class is not 'crossover', 'junction', or 'terminal', zero out the bbox region
            if self.debug:
                print(f"EmptiedMask Log: Processing bbox class: {bbox['class']}, UID: {bbox.get('persistent_uid')}")
            
           
            components_to_preserve_in_mask = ('crossover', 'junction', 'circuit', 'vss') 

            if bbox['class'] not in components_to_preserve_in_mask:
                if self.debug:
                    print(f"EmptiedMask Log: Zeroing out {bbox['class']} (UID: {bbox.get('persistent_uid')})")
                image_copy[int(bbox['ymin']):int(bbox['ymax']), 
                          int(bbox['xmin']):int(bbox['xmax'])] = 0
            else:
                if self.debug:
                    print(f"EmptiedMask Log: PRESERVING {bbox['class']} (UID: {bbox.get('persistent_uid')}) in mask")
                
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
    
    def _are_bboxes_proximal_for_clustering(self, bbox1_dict_or_tuple, bbox2_dict_or_tuple, proximity_threshold=50):
        """
        Checks if two bounding boxes are proximal for clustering purposes.
        Similar to are_bboxes_proximal in enumerate_components but used for crop clustering.
        """
        if isinstance(bbox1_dict_or_tuple, dict):
            xmin1, ymin1, xmax1, ymax1 = bbox1_dict_or_tuple['xmin'], bbox1_dict_or_tuple['ymin'], bbox1_dict_or_tuple['xmax'], bbox1_dict_or_tuple['ymax']
        else: # tuple
            xmin1, ymin1, xmax1, ymax1 = bbox1_dict_or_tuple

        if isinstance(bbox2_dict_or_tuple, dict):
            xmin2, ymin2, xmax2, ymax2 = bbox2_dict_or_tuple['xmin'], bbox2_dict_or_tuple['ymin'], bbox2_dict_or_tuple['xmax'], bbox2_dict_or_tuple['ymax']
        else: # tuple
            xmin2, ymin2, xmax2, ymax2 = bbox2_dict_or_tuple

        # Check for direct overlap first
        if not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2):
            return True # They overlap

        # Check proximity (distance between closest edges)
        # Horizontal distance
        if xmax1 < xmin2: # bbox1 is to the left of bbox2
            h_dist = xmin2 - xmax1
        elif xmin1 > xmax2: # bbox1 is to the right of bbox2
            h_dist = xmin1 - xmax2
        else: # Overlap or aligned vertically
            h_dist = 0
        
        # Vertical distance
        if ymax1 < ymin2: # bbox1 is above bbox2
            v_dist = ymin2 - ymax1
        elif ymin1 > ymax2: # bbox1 is below bbox2
            v_dist = ymin1 - ymax2
        else: # Overlap or aligned horizontally
            v_dist = 0
        
        return h_dist <= proximity_threshold and v_dist <= proximity_threshold

    def _component_has_nearby_text(self, component_bbox, text_bboxes, proximity_threshold=30):
        """Checks if a component bounding box has any text bounding box nearby."""
        for text_bbox in text_bboxes:
            if self._are_bboxes_proximal_for_clustering(component_bbox, text_bbox, proximity_threshold):
                return True
        return False

    def crop_image_and_adjust_bboxes(self, image_to_crop, all_yolo_bboxes_input, padding=20):
        """
        Crops an image based on a defining bounding box (derived from YOLO component clusters)
        and adjusts other bounding boxes. Preserves persistent_uid on adjusted bboxes.

        Args:
            image_to_crop (np.ndarray): The original image to be cropped.
            all_yolo_bboxes_input (List[Dict]): List of all YOLO bboxes in original image coordinates.
            padding (int): Padding around the determined crop basis bbox.

        Returns:
            Tuple[np.ndarray, List[Dict], Dict | None]:
                - Cropped image (or original if no crop is performed).
                - New list of adjusted bboxes (or original if no crop).
                - crop_debug_info (Dict): A dictionary containing detailed information about the crop operation.
        """
        original_height, original_width = image_to_crop.shape[:2]
        crop_debug_info = {
            'crop_applied': False,
            'reason_for_no_crop': None,
            'original_image_dims': (original_width, original_height),
            'num_total_yolo_bboxes': len(all_yolo_bboxes_input),
            'num_component_type_bboxes': 0,
            'num_text_type_bboxes': 0,
            'clustering_proximity_threshold': None,
            'num_clusters_found': None,
            'main_cluster_info': None,
            'crop_decision_source': "unknown",
            'crop_basis_bbox_before_padding': None,
            'padding_value': padding,
            'window_after_main_padding': None,
            'text_bboxes_that_expanded_crop': [],
            'final_crop_window_abs': None,
            'cropped_image_dims': (original_width, original_height) # Default to original
        }

        # 1. Filter bboxes
        component_type_bboxes = [
            b for b in all_yolo_bboxes_input 
            if b.get('class') not in self.non_components 
        ]
        text_type_bboxes = [b for b in all_yolo_bboxes_input if b.get('class') == 'text']
        crop_debug_info['num_component_type_bboxes'] = len(component_type_bboxes)
        crop_debug_info['num_text_type_bboxes'] = len(text_type_bboxes)
        
        elements_for_clustering = [
        b for b in all_yolo_bboxes_input 
        if b.get('class') not in {'text', 'explanatory', 'circuit', 'vss', 'crossover'} 
        ]
        if self.debug:
            print(f"Crop: Number of elements for clustering (components + junctions): {len(elements_for_clustering)}")

        crop_basis_bbox = None # This will be (xmin, ymin, xmax, ymax) tuple

        if not elements_for_clustering: # MODIFIED from component_type_bboxes
            if self.debug: print("Crop: No elements_for_clustering (components or junctions) found. No basis for crop.") 
            crop_debug_info['reason_for_no_crop'] = "no_elements_for_clustering" 
            crop_debug_info['crop_decision_source'] = "no_crop_due_to_no_clustering_elements" 
            return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info
        else:
            # Build adjacency list for elements_for_clustering 
            adj = {i: [] for i in range(len(elements_for_clustering))} 
            
            # Dynamic proximity threshold for clustering
            # Calculate average component size for a more adaptive threshold
            non_junction_elements_for_sizing = [el for el in elements_for_clustering if el.get('class') != 'junction']
            avg_diag = 0 # Initialize avg_diag
            if non_junction_elements_for_sizing:
                avg_w = sum(b['xmax'] - b['xmin'] for b in non_junction_elements_for_sizing) / len(non_junction_elements_for_sizing)
                avg_h = sum(b['ymax'] - b['ymin'] for b in non_junction_elements_for_sizing) / len(non_junction_elements_for_sizing)
                avg_diag = np.sqrt(avg_w**2 + avg_h**2)
                clustering_prox_threshold = max(int(avg_diag * 2.0), 30) # MODIFIED: Multiplier 2.0, min 30
                if self.debug:
                    print(f"Crop: Avg NON-JUNCTION component w: {avg_w:.2f}, h: {avg_h:.2f}, diag: {avg_diag:.2f}")
                    print(f"Crop: Calculated adaptive clustering_prox_threshold (non-junctions): {clustering_prox_threshold}px")
            elif elements_for_clustering: # Fallback: if only junctions are available for sizing
                avg_w = sum(b['xmax'] - b['xmin'] for b in elements_for_clustering) / len(elements_for_clustering)
                avg_h = sum(b['ymax'] - b['ymin'] for b in elements_for_clustering) / len(elements_for_clustering)
                avg_diag = np.sqrt(avg_w**2 + avg_h**2)
                clustering_prox_threshold = max(int(avg_diag * 2.5), 20) # MODIFIED: Min 20 for junction-only
                if self.debug:
                    print(f"Crop: Only junctions found for sizing. Avg junction w: {avg_w:.2f}, h: {avg_h:.2f}, diag: {avg_diag:.2f}")
                    print(f"Crop: Calculated adaptive clustering_prox_threshold (junctions only): {clustering_prox_threshold}px")
            else:
                # This case should ideally not be reached if elements_for_clustering is not empty
                clustering_prox_threshold = 50 # Absolute fallback 
            
            crop_debug_info['clustering_proximity_threshold'] = clustering_prox_threshold

            for i in range(len(elements_for_clustering)): 
                for j in range(i + 1, len(elements_for_clustering)): 
                    if self._are_bboxes_proximal_for_clustering(elements_for_clustering[i], elements_for_clustering[j], proximity_threshold=clustering_prox_threshold): 
                        adj[i].append(j)
                        adj[j].append(i)

            # Find connected components (DFS)
            visited_indices = [False] * len(elements_for_clustering) 
            clusters_of_bboxes_list = [] # List of lists of bbox dicts
            for i in range(len(elements_for_clustering)): 
                if not visited_indices[i]:
                    current_cluster_member_bboxes = []
                    component_indices_in_stack = [i]
                    while component_indices_in_stack:
                        u_idx = component_indices_in_stack.pop()
                        if not visited_indices[u_idx]:
                            visited_indices[u_idx] = True
                            current_cluster_member_bboxes.append(elements_for_clustering[u_idx]) 
                            # Iterate over neighbors from adjacency list
                            for v_neighbor_idx in adj[u_idx]:
                                if not visited_indices[v_neighbor_idx]:
                                    component_indices_in_stack.append(v_neighbor_idx)
                    if current_cluster_member_bboxes:
                        clusters_of_bboxes_list.append(current_cluster_member_bboxes)
            
            crop_debug_info['num_clusters_found'] = len(clusters_of_bboxes_list)

            if not clusters_of_bboxes_list:
                if self.debug: print("Crop: Elements for clustering found, but no clusters formed. Using union of all elements.") 
                min_x = min(b['xmin'] for b in elements_for_clustering) 
                min_y = min(b['ymin'] for b in elements_for_clustering) 
                max_x = max(b['xmax'] for b in elements_for_clustering) 
                max_y = max(b['ymax'] for b in elements_for_clustering) 
                crop_basis_bbox = (min_x, min_y, max_x, max_y)
                crop_debug_info['crop_decision_source'] = "union_of_isolated_elements_for_clustering" 
                crop_debug_info['main_cluster_info'] = "all_elements_isolated_used_union" 
            else:
                # Refined main cluster selection:
                # Score clusters by component count and text association.
                # Use avg_diag from the sizing calculation (non-junction preferred, initialized to 0 if no elements for sizing)
                current_avg_diag_for_text_prox = avg_diag if avg_diag > 0 else 30 # Fallback for text prox if avg_diag is 0
                text_association_proximity_for_cluster_scoring = max(int(current_avg_diag_for_text_prox * 0.75), 25)
                
                scored_clusters = []
                for i, cluster_candidate_bboxes in enumerate(clusters_of_bboxes_list):
                    text_associated_components_in_cluster = 0
                    # Only count text association for actual components, not junctions
                    actual_components_in_this_cluster = [b for b in cluster_candidate_bboxes if b.get('class') != 'junction']
                    for comp_b in actual_components_in_this_cluster: # Iterate over actual components
                        if self._component_has_nearby_text(comp_b, text_type_bboxes, text_association_proximity_for_cluster_scoring):
                            text_associated_components_in_cluster += 1
                    
                    # Score: primary by num text-associated (actual) components, secondary by total elements in cluster
                    score = (text_associated_components_in_cluster, len(cluster_candidate_bboxes))
                    scored_clusters.append({
                        'bboxes': cluster_candidate_bboxes, 
                        'score': score, 
                        'id': i,
                        'text_assoc_count': text_associated_components_in_cluster,
                        'total_elements_in_cluster': len(cluster_candidate_bboxes), # New field
                        'actual_components_in_cluster': len(actual_components_in_this_cluster), # New field
                        })
                    if self.debug:
                        print(f"Crop: Cluster {i} - Score: {score}, TextAssocActualComps: {text_associated_components_in_cluster}, TotalElements: {len(cluster_candidate_bboxes)}, ActualComps: {len(actual_components_in_this_cluster)}")

                # Sort clusters: highest score first
                scored_clusters.sort(key=lambda c: c['score'], reverse=True)
                
                main_cluster_actual_bboxes = [] # Initialize to ensure it's defined
                main_cluster_info_log = {} # Initialize for logging

                if not scored_clusters: # Should not happen if clusters_of_bboxes_list was not empty
                    if self.debug: print("Crop: ERROR - scored_clusters is empty. Defaulting to largest raw cluster by len.")
                    if clusters_of_bboxes_list: # Ensure clusters_of_bboxes_list itself isn't empty
                        main_cluster_actual_bboxes = max(clusters_of_bboxes_list, key=len)
                        crop_debug_info['crop_decision_source'] = "main_cluster_fallback_scoring_failed_used_max_len"
                        main_cluster_info_log = {'num_elements': len(main_cluster_actual_bboxes), 'reason': 'scoring_failed_default_max_len'}
                    else: # This means elements_for_clustering existed, but formed no clusters, and clusters_of_bboxes_list is empty - this path should be covered by the earlier "if not clusters_of_bboxes_list:" block
                        if self.debug: print("Crop: CRITICAL - elements_for_clustering existed, but no clusters AND scored_clusters empty. No basis for crop.")
                        crop_debug_info['reason_for_no_crop'] = "no_clusters_formed_and_scoring_empty"
                        return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info
                elif scored_clusters[0]['text_assoc_count'] == 0 and scored_clusters[0]['actual_components_in_cluster'] > 0:
                    if self.debug: 
                        print("Crop: Best cluster has >0 actual components but 0 have associated text. Falling back to largest cluster by *total element count* from all original clusters.")
                    main_cluster_actual_bboxes = max(clusters_of_bboxes_list, key=len) 
                    crop_debug_info['crop_decision_source'] = "main_cluster_fallback_no_text_assoc_in_best_with_components"
                    # Ensure selected_cluster_info_for_log is robustly assigned
                    selected_cluster_info_for_log = next((c for c in scored_clusters if c['bboxes'] == main_cluster_actual_bboxes), None) # Find the actual selected cluster
                    if selected_cluster_info_for_log is None: # If not found (e.g. main_cluster_actual_bboxes was empty or not in scored_clusters list for some reason)
                        selected_cluster_info_for_log = scored_clusters[0] if scored_clusters else {} # Fallback to best or empty dict
                    
                    main_cluster_info_log = {
                        'num_elements': len(main_cluster_actual_bboxes),
                        'text_assoc_count': selected_cluster_info_for_log.get('text_assoc_count', -1), # Use .get for safety
                        'score': selected_cluster_info_for_log.get('score', (-1,-1)), 
                        'id': selected_cluster_info_for_log.get('id', -1)
                    }
                else: # Normal case or best cluster has no actual components (e.g. only junctions, text_assoc_count will be 0)
                    main_cluster_actual_bboxes = scored_clusters[0]['bboxes']
                    selected_cluster_info = scored_clusters[0]
                    main_cluster_info_log = {
                        'num_elements': len(main_cluster_actual_bboxes),
                        'text_assoc_count': selected_cluster_info['text_assoc_count'],
                        'score': selected_cluster_info['score'],
                        'id': selected_cluster_info['id']
                    }
                    crop_debug_info['crop_decision_source'] = "main_yolo_cluster_scored_by_text_assoc"
                
                if main_cluster_actual_bboxes: # Add example_uid if cluster is not empty
                    main_cluster_info_log['example_uid'] = main_cluster_actual_bboxes[0].get('persistent_uid')
                else: # Safety if somehow main_cluster_actual_bboxes ended up empty
                    main_cluster_info_log['example_uid'] = "N/A_EMPTY_CLUSTER_SELECTED"
                    if self.debug: print("Crop: WARNING - main_cluster_actual_bboxes is empty after selection logic.")
                
                crop_debug_info['main_cluster_info'] = main_cluster_info_log
                
                if not main_cluster_actual_bboxes: # Final safety check before min/max
                    if self.debug: print("Crop: CRITICAL - main_cluster_actual_bboxes is EMPTY before min/max calculation. Returning originals.")
                    crop_debug_info['reason_for_no_crop'] = "main_cluster_empty_before_min_max"
                    return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info

                mc_min_x = min(b['xmin'] for b in main_cluster_actual_bboxes)
                mc_min_y = min(b['ymin'] for b in main_cluster_actual_bboxes)
                mc_max_x = max(b['xmax'] for b in main_cluster_actual_bboxes)
                mc_max_y = max(b['ymax'] for b in main_cluster_actual_bboxes)
                crop_basis_bbox = (mc_min_x, mc_min_y, mc_max_x, mc_max_y)
                # crop_debug_info['crop_decision_source'] is set by specific branches

        if crop_basis_bbox is None:
            if self.debug: print("Crop: crop_basis_bbox is None after YOLO processing. No viable basis for cropping. Returning originals.")
            crop_debug_info['reason_for_no_crop'] = "no_viable_yolo_crop_basis_found"
            return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info

        crop_debug_info['crop_basis_bbox_before_padding'] = crop_basis_bbox
        
        # Use crop_basis_bbox as the defining extent for padding and text inclusion
        def_xmin, def_ymin, def_xmax, def_ymax = crop_basis_bbox

        # Early exit if the crop_basis_bbox is already very large
        original_area = float(original_height * original_width)
        basis_bbox_width = float(max(0, def_xmax - def_xmin))
        basis_bbox_height = float(max(0, def_ymax - def_ymin))
        basis_bbox_area = basis_bbox_width * basis_bbox_height

        if original_area > 0 and (basis_bbox_area / original_area) > 0.90:
            if self.debug:
                print(f"Crop: Crop basis bbox area ({basis_bbox_area:.0f}) is > 90% of original image area ({original_area:.0f}). Skipping crop.")
            crop_debug_info['reason_for_no_crop'] = "crop_basis_bbox_too_large"
            return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info

        # Initial crop window calculation based on crop_basis_bbox and padding
        current_crop_xmin = float(max(0, def_xmin - padding))
        current_crop_ymin = float(max(0, def_ymin - padding))
        current_crop_xmax = float(min(original_width, def_xmax + padding))
        current_crop_ymax = float(min(original_height, def_ymax + padding))
        crop_debug_info['window_after_main_padding'] = (int(round(current_crop_xmin)), int(round(current_crop_ymin)), int(round(current_crop_xmax)), int(round(current_crop_ymax)))
        
        if self.debug:
            print(f"Crop: Window after crop_basis_bbox and main padding: x({current_crop_xmin:.0f}-{current_crop_xmax:.0f}), y({current_crop_ymin:.0f}-{current_crop_ymax:.0f})")

        # Expand crop window to ensure sufficient padding for 'text' boxes
        text_inclusion_padding = 20 

        for text_bbox_item in text_type_bboxes: # Iterate over actual text bboxes
            text_xmin, text_ymin, text_xmax, text_ymax = float(text_bbox_item['xmin']), float(text_bbox_item['ymin']), float(text_bbox_item['xmax']), float(text_bbox_item['ymax'])
            
            # Check if the text box is reasonably close to the current crop window before expanding for it.
            # This prevents distant text boxes from massively expanding the crop.
            # For example, check if text_bbox_item overlaps or is near current_crop_xmin,ymin,xmax,ymax
            # A simple check: if text box is entirely outside a slightly expanded version of current crop window, ignore it.
            expanded_check_padding = 150 # Check against a slightly larger current window
            if not (text_xmax < current_crop_xmin - expanded_check_padding or \
                    text_xmin > current_crop_xmax + expanded_check_padding or \
                    text_ymax < current_crop_ymin - expanded_check_padding or \
                    text_ymin > current_crop_ymax + expanded_check_padding):

                expanded_xmin_for_text = min(current_crop_xmin, max(0, text_xmin - text_inclusion_padding))
                expanded_ymin_for_text = min(current_crop_ymin, max(0, text_ymin - text_inclusion_padding))
                expanded_xmax_for_text = max(current_crop_xmax, min(original_width, text_xmax + text_inclusion_padding))
                expanded_ymax_for_text = max(current_crop_ymax, min(original_height, text_ymax + text_inclusion_padding))

                did_expand = (expanded_xmin_for_text != current_crop_xmin or 
                              expanded_ymin_for_text != current_crop_ymin or 
                              expanded_xmax_for_text != current_crop_xmax or 
                              expanded_ymax_for_text != current_crop_ymax)

                current_crop_xmin, current_crop_ymin, current_crop_xmax, current_crop_ymax = expanded_xmin_for_text, expanded_ymin_for_text, expanded_xmax_for_text, expanded_ymax_for_text
                
                if did_expand:
                    if self.debug:
                        print(f"Crop: Text box UID {text_bbox_item.get('persistent_uid')} at ({text_xmin:.0f},{text_ymin:.0f})-({text_xmax:.0f},{text_ymax:.0f}) ensured text_inclusion_padding.")
                        print(f"Crop: Updated crop window for text padding: x({current_crop_xmin:.0f}-{current_crop_xmax:.0f}), y({current_crop_ymin:.0f}-{current_crop_ymax:.0f})")
                    crop_debug_info['text_bboxes_that_expanded_crop'].append({
                        'uid': text_bbox_item.get('persistent_uid'),
                        'class': text_bbox_item.get('class'),
                        'coords_original': (text_bbox_item['xmin'], text_bbox_item['ymin'], text_bbox_item['xmax'], text_bbox_item['ymax']),
                        'coords_text_box_abs': (text_xmin, text_ymin, text_xmax, text_ymax)
                    })
            elif self.debug:
                print(f"Crop: Text box UID {text_bbox_item.get('persistent_uid')} was too far, not used for expansion.")


        # Finalize crop coordinates as integers, ensuring they remain within image boundaries
        crop_abs_xmin = max(0, int(round(current_crop_xmin)))
        crop_abs_ymin = max(0, int(round(current_crop_ymin)))
        crop_abs_xmax = min(original_width, int(round(current_crop_xmax)))
        crop_abs_ymax = min(original_height, int(round(current_crop_ymax)))
        crop_debug_info['final_crop_window_abs'] = (crop_abs_xmin, crop_abs_ymin, crop_abs_xmax, crop_abs_ymax)

        if self.debug:
            print(f"Crop: Original image dims: {original_width}x{original_height}")
            print(f"Crop: Crop_basis_bbox (e.g. from YOLO cluster): {crop_basis_bbox}")
            print(f"Crop: Final calculated absolute crop window (xmin,ymin,xmax,ymax): {crop_abs_xmin}, {crop_abs_ymin}, {crop_abs_xmax}, {crop_abs_ymax}")

        if crop_abs_xmin >= crop_abs_xmax or crop_abs_ymin >= crop_abs_ymax:
            if self.debug:
                print("Crop: Invalid crop region after all calculations. Returning originals.")
            crop_debug_info['reason_for_no_crop'] = "invalid_region_after_expansion"
            return image_to_crop, [deepcopy(b) for b in all_yolo_bboxes_input], crop_debug_info

        cropped_image = image_to_crop[crop_abs_ymin:crop_abs_ymax, crop_abs_xmin:crop_abs_xmax]
        new_height_cr, new_width_cr = cropped_image.shape[:2] # Renamed to avoid conflict
        crop_debug_info['cropped_image_dims'] = (new_width_cr, new_height_cr)
        crop_debug_info['crop_applied'] = True

        if self.debug:
            print(f"Crop: Cropped image shape: {cropped_image.shape}")

        adjusted_bboxes = []
        for bbox_original in all_yolo_bboxes_input: # Adjust all original bboxes
            adj_bbox = deepcopy(bbox_original) 
            adj_bbox['xmin'] = bbox_original['xmin'] - crop_abs_xmin
            adj_bbox['ymin'] = bbox_original['ymin'] - crop_abs_ymin
            adj_bbox['xmax'] = bbox_original['xmax'] - crop_abs_xmin
            adj_bbox['ymax'] = bbox_original['ymax'] - crop_abs_ymin

            # Clip adjusted bboxes to the dimensions of the new cropped image
            adj_bbox['xmin'] = max(0, adj_bbox['xmin'])
            adj_bbox['ymin'] = max(0, adj_bbox['ymin'])
            adj_bbox['xmax'] = min(new_width_cr, adj_bbox['xmax'])
            adj_bbox['ymax'] = min(new_height_cr, adj_bbox['ymax'])

            # Only keep bboxes that still have a positive area after adjustment and clipping
            if adj_bbox['xmax'] > adj_bbox['xmin'] and adj_bbox['ymax'] > adj_bbox['ymin']:
                adjusted_bboxes.append(adj_bbox)
            elif self.debug:
                print(f"Crop: Bbox for class {adj_bbox.get('class')} (UID: {adj_bbox.get('persistent_uid')}) filtered out post-crop. Orig coords: ({bbox_original['xmin']},{bbox_original['ymin']})-({bbox_original['xmax']},{bbox_original['ymax']}). Adjusted before clip: ({bbox_original['xmin'] - crop_abs_xmin},{bbox_original['ymin'] - crop_abs_ymin}). Clipped: ({adj_bbox['xmin']},{adj_bbox['ymin']})-({adj_bbox['xmax']},{adj_bbox['ymax']})")
        
        if self.debug:
            print(f"Crop: Original bbox count: {len(all_yolo_bboxes_input)}, Adjusted bbox count: {len(adjusted_bboxes)}")
        
        return cropped_image, adjusted_bboxes, crop_debug_info

    def get_node_connections(self, _image_for_context, processing_wire_mask, bboxes_relative_to_mask):
        if self.debug:
            print(f"Node Connections: Received processing_wire_mask shape: {processing_wire_mask.shape if processing_wire_mask is not None else 'None'}, {len(bboxes_relative_to_mask)} bboxes.")

        # Graceful exit if processing_wire_mask is None (e.g., SAM2 disabled or failed)
        if processing_wire_mask is None:
            if self.debug:
                print("Node Connections: processing_wire_mask is None. Cannot perform node analysis. Returning empty results.")
            # Return structures matching expected output, but empty/default where appropriate
            # The visualization images would typically be based on the input image or a default size if mask is None.
            # For simplicity, we can return None for image outputs or a small blank image if a placeholder is strictly needed.
            
            # Fallback image for visualizations if processing_wire_mask was expected for dimensions
            fallback_viz_height, fallback_viz_width = (100,100) # Small default
            if _image_for_context is not None: # If original image context is available, use its dims for fallback viz
                fallback_viz_height, fallback_viz_width = _image_for_context.shape[:2]

            blank_image_fallback = np.zeros((fallback_viz_height, fallback_viz_width, 3), dtype=np.uint8)
            
            return [], blank_image_fallback, blank_image_fallback, blank_image_fallback, blank_image_fallback, blank_image_fallback

        # --- Mask Preparation (Input `processing_wire_mask` is already the cropped SAM2 mask) ---
        emptied_mask = processing_wire_mask.copy()
        current_height, current_width = emptied_mask.shape[:2]

        if self.debug:
            # Find the specific terminal bbox for detailed logging
            problematic_terminal_uid = "terminal_333_266_410_334" # As seen in logs
            problematic_bbox_for_log = None
            for bbox_check in bboxes_relative_to_mask:
                if bbox_check.get('persistent_uid') == problematic_terminal_uid:
                    problematic_bbox_for_log = bbox_check
                    break
            
            if problematic_bbox_for_log:
                bb_ymin, bb_ymax = max(0, int(problematic_bbox_for_log['ymin'])), min(current_height, int(problematic_bbox_for_log['ymax']))
                bb_xmin, bb_xmax = max(0, int(problematic_bbox_for_log['xmin'])), min(current_width, int(problematic_bbox_for_log['xmax']))
                if bb_ymin < bb_ymax and bb_xmin < bb_xmax:
                    terminal_area_in_processing_mask = processing_wire_mask[bb_ymin:bb_ymax, bb_xmin:bb_xmax]
                    is_terminal_present_in_processing_mask = np.any(terminal_area_in_processing_mask)
                    print(f"NodeConnLog: Problematic terminal UID {problematic_terminal_uid} area in INPUT processing_wire_mask. Is present: {is_terminal_present_in_processing_mask}. Sum: {np.sum(terminal_area_in_processing_mask)}")

        for bbox_comp in bboxes_relative_to_mask:
            # MODIFIED: Log decision for the problematic terminal
            is_problematic_terminal_current_bbox = bbox_comp.get('persistent_uid') == problematic_terminal_uid
            
            components_to_preserve_locally_in_mask = ('crossover', 'junction', 'circuit', 'vss') 
            
            if bbox_comp['class'] not in components_to_preserve_locally_in_mask:
                if self.debug and is_problematic_terminal_current_bbox:
                    print(f"NodeConnLog: Problematic terminal UID {problematic_terminal_uid} (class: {bbox_comp['class']}) is NOT in components_to_preserve_locally_in_mask. Will be ZEROED OUT by this loop.")
                
                ymin, ymax = max(0, int(bbox_comp['ymin'])), min(current_height, int(bbox_comp['ymax']))
                xmin, xmax = max(0, int(bbox_comp['xmin'])), min(current_width, int(bbox_comp['xmax']))
                if ymin < ymax and xmin < xmax:
                    emptied_mask[ymin:ymax, xmin:xmax] = 0
            else: # Component IS in components_to_preserve_locally
                if self.debug and is_problematic_terminal_current_bbox:
                    print(f"NodeConnLog: Problematic terminal UID {problematic_terminal_uid} (class: {bbox_comp['class']}) IS IN components_to_preserve_locally. Will be PRESERVED by this loop.")
                # No action, area is preserved
        
        if self.debug:
            self.show_image(emptied_mask, 'Emptied Mask (from cropped SAM2, pre-resize, after NodeConn local empty)')
            # Log state of problematic terminal in the locally emptied_mask
            if problematic_bbox_for_log: # Check if found earlier
                bb_ymin, bb_ymax = max(0, int(problematic_bbox_for_log['ymin'])), min(current_height, int(problematic_bbox_for_log['ymax']))
                bb_xmin, bb_xmax = max(0, int(problematic_bbox_for_log['xmin'])), min(current_width, int(problematic_bbox_for_log['xmax']))
                if bb_ymin < bb_ymax and bb_xmin < bb_xmax:
                    terminal_area_in_local_emptied_mask = emptied_mask[bb_ymin:bb_ymax, bb_xmin:bb_xmax]
                    is_terminal_present_in_local_emptied_mask = np.any(terminal_area_in_local_emptied_mask)
                    print(f"NodeConnLog: Problematic terminal UID {problematic_terminal_uid} area in LOCAL emptied_mask (AFTER loop). Is present: {is_terminal_present_in_local_emptied_mask}. Sum: {np.sum(terminal_area_in_local_emptied_mask)}")


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
                    
                    # Determine pixel threshold based on component type
                    current_pixel_threshold = 6 # Default
                    component_class_for_threshold = bbox_comp_proc_resized['class']
                    # self.source_components is defined in __init__
                    # Check if component_class_for_threshold is in the set of source components
                    if component_class_for_threshold in self.source_components: 
                        current_pixel_threshold = 20 # Larger threshold for sources (increased from 10 to 20)
                    # Check if component_class_for_threshold is in a list of other sensitive components
                    elif component_class_for_threshold in ['diode', 'diode.light_emitting', 'diode.zener', 'transistor.bjt', 'transistor.fet']:
                        current_pixel_threshold = 8

                    # is_point_near_bbox expects bbox_comp_proc_resized (already in resized system)
                    if self.is_point_near_bbox(point, bbox_comp_proc_resized, pixel_threshold=current_pixel_threshold):
                        # IMPORTANT: Store the component from processing_bboxes_resized to align coordinate systems
                        # for geometric reasoning with node contours. persistent_uid and semantic_direction
                        # should have been copied by resize_bboxes.
                        target_bbox_to_add = deepcopy(processing_bboxes_resized[i])
                        
                        component_unique_ref = target_bbox_to_add.get('persistent_uid')
                        # Fallback if persistent_uid is somehow missing (should not happen with current crop logic)
                        if component_unique_ref is None: 
                             if self.debug: 
                                print(f"WARNING: persistent_uid missing for component: {target_bbox_to_add.get('class')}")
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
            if len(viz_fallback.shape) == 2:
                viz_fallback = cv2.cvtColor(viz_fallback, cv2.COLOR_GRAY2BGR)
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
            if self.debug:
                print("Node Conn: Ground node ID not properly determined. Arbitrary numbering for all valid nodes.")
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

        
        # Process components
        for n_idx, n_data in enumerate(node_list):
            node_components = n_data['components']
            current_node_id = n_data['id']
            
            for component in node_components:
                component_class = component.get('class') # This is the class after potential reclassification
                persistent_uid = component.get('persistent_uid')
                
                # Semantic direction and reason would have been added by LLaMA if the component was eligible
                component_semantic_direction = component.get('semantic_direction', 'UNKNOWN')
                semantic_reason = component.get('semantic_reason', 'UNKNOWN')

                if not persistent_uid:
                    if self.debug:
                        print(f"Warning: Component {component_class} missing persistent_uid. Skipping.")
                    continue

                is_ignorable_class = component_class in ['text', 'explanatory', 'junction', 'crossover']
                is_already_processed = persistent_uid in processed_components
                if is_ignorable_class or is_already_processed:
                    continue
                processed_components.add(persistent_uid)

                other_node_id = None
                for other_n_data in node_list:
                    if other_n_data['id'] != current_node_id:
                        if any(c.get('persistent_uid') == persistent_uid for c in other_n_data['components']):
                            other_node_id = other_n_data['id']
                            break
                
                # REVISED LOGIC BLOCK
                current_yolo_class = component.get('class') # Class after potential reclassification

                if current_yolo_class == 'terminal': 
                    # This means it was 'terminal' and prelim check found < 2 connections
                    if self.debug:
                        print(f"NetlistGen: Component UID {persistent_uid} is still 'terminal'. Classifying as type 'N'.")
                    component_type_prefix = self.netlist_map.get('terminal', 'N')
                    node_1 = current_node_id
                    node_2 = '0'  # Default other node to ground for a true terminal type 'N'
                    value = "None"
                else: 
                    # Component is not 'terminal' (either originally, or reclassified from terminal e.g. to 'voltage.dc')
                    if other_node_id is None:
                        # This case should be rare if reclassification worked, but as a safeguard
                        if self.debug:
                            print(f"Warning: Non-terminal component UID {persistent_uid} ({current_yolo_class}) has no second node. Skipping.")
                        continue
                    
                    # Determine component_type_prefix. 
                    # LLaMA would have run on the current_yolo_class if it was eligible.
                    prospective_prefix = self.netlist_map.get(current_yolo_class, 'UN')
                    reason_for_prefix = component.get('semantic_reason', 'UNKNOWN') # From LLaMA output

                    # Apply LLaMA-based prefix adjustments if applicable
                    if current_yolo_class in self.voltage_classes_names and reason_for_prefix == "ARROW":
                        prospective_prefix = 'I'
                    elif current_yolo_class in self.current_source_classes_names and reason_for_prefix == "SIGN":
                        prospective_prefix = 'V'
                    component_type_prefix = prospective_prefix

                    if not component_type_prefix:
                        if self.debug:
                            print(f"Warning: Component {current_yolo_class} (UID: {persistent_uid}) resulted in empty component_type_prefix. Skipping.")
                        continue

                    # Node ordering using centroids and LLaMA-derived semantic direction
                    current_node_centroid_val = node_centroids.get(current_node_id)
                    other_node_centroid_val = node_centroids.get(other_node_id)

                    if current_node_centroid_val is None or other_node_centroid_val is None:
                        if self.debug:
                            print(f"Warning: Missing centroid for nodes ({current_node_id}, {other_node_id}) for UID {persistent_uid}. Arbitrary node order.")
                        assigned_node1_id, assigned_node2_id = current_node_id, other_node_id
                    else:
                        # component_semantic_direction is from the bbox, potentially set by LLaMA
                        primary_centroid, secondary_centroid = self._get_terminal_nodes_relative_to_bbox(
                            component, 
                            component_semantic_direction, 
                            current_node_centroid_val, 
                            other_node_centroid_val, 
                            current_yolo_class, # Use the current class (potentially reclassified)
                            semantic_reason     # Use reason from LLaMA
                        )
                        if primary_centroid == current_node_centroid_val:
                            assigned_node1_id, assigned_node2_id = current_node_id, other_node_id
                        else:
                            assigned_node1_id, assigned_node2_id = other_node_id, current_node_id
                    
                    # Final node assignment (handling gnd)
                    if current_yolo_class in ['gnd', 'vss']:
                        if assigned_node1_id == 0:
                            true_node = assigned_node2_id
                        else:
                            true_node = assigned_node1_id
                        node_1, node_2 = true_node, 0
                    else:
                        node_1 = assigned_node1_id
                        node_2 = assigned_node2_id
                    value = "None" # Placeholder for VLM
                # END OF REVISED LOGIC BLOCK
                
                if not component_type_prefix: # Safeguard: if prefix ended up empty, skip
                    if self.debug:
                        print(f"NetlistGen: Skipping component UID {persistent_uid} due to empty final component_type_prefix.")
                    continue

                # Get unique component ID (R1, C1, etc.)
                if component_type_prefix not in component_counters:
                     component_counters[component_type_prefix] = 1
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
        # First Pass: Update component information (class, type, value) from VLM output
        # and store visual_id for sorting.
        for line_from_gen_netlist in netlist:
            target_persistent_uid = line_from_gen_netlist.get('persistent_uid')
            if not target_persistent_uid:
                if self.debug:
                    print(f"Debug fix_netlist (Pass 1): Line in netlist missing persistent_uid: {line_from_gen_netlist}")
                continue

            visual_id_for_this_component = None
            for enum_bbox in all_enumerated_bboxes:
                if enum_bbox.get('persistent_uid') == target_persistent_uid:
                    visual_id_for_this_component = enum_bbox.get('id')
                    break
            
            line_from_gen_netlist['visual_id'] = visual_id_for_this_component # Store for sorting

            if visual_id_for_this_component is None:
                if self.debug:
                    print(f"Debug fix_netlist (Pass 1): Could not find visual_id for persistent_uid {target_persistent_uid}. Will not be updated by VLM.")
                # Ensure 'class' and 'component_type' are consistent if no VLM update
                if 'class' not in line_from_gen_netlist: # Should exist from generate_netlist
                    line_from_gen_netlist['class'] = 'unknown' # Fallback
                if 'component_type' not in line_from_gen_netlist: # Should exist
                    line_from_gen_netlist['component_type'] = self.netlist_map.get(line_from_gen_netlist['class'], 'UN')
                continue

            found_vlm_match = False
            for vlm_item in vlm_out:
                if str(vlm_item.get('id')) == str(visual_id_for_this_component):
                    found_vlm_match = True
                    current_value_in_netlist_line = line_from_gen_netlist.get('value')
                    vlm_provided_value = vlm_item.get('value')
                    effective_vlm_value = vlm_provided_value
                    
                    vlm_class = vlm_item.get('class')
                    if not vlm_class: # If VLM provides no class, skip VLM update for this item.
                        if self.debug:
                            print(f"Debug fix_netlist (Pass 1): VLM item for visual_id {visual_id_for_this_component} has no class. Skipping VLM update for this component.")
                        # Ensure 'class' and 'component_type' are consistent if no VLM update
                        if 'class' not in line_from_gen_netlist:
                            line_from_gen_netlist['class'] = 'unknown'
                        if 'component_type' not in line_from_gen_netlist:
                            line_from_gen_netlist['component_type'] = self.netlist_map.get(line_from_gen_netlist['class'], 'UN')
                        break 

                    prospective_component_type_from_vlm = self.netlist_map.get(vlm_class, 'UN')

                    if prospective_component_type_from_vlm in ['V', 'I']:
                        if isinstance(vlm_provided_value, str):
                            try:
                                float(vlm_provided_value)
                            except ValueError:
                                if vlm_provided_value.isalpha() and vlm_provided_value.lower() != 'ac':
                                    if self.debug:
                                        print(f"Debug fix_netlist (Pass 1): Independent source {prospective_component_type_from_vlm} (VLM class {vlm_class}) with problematic alpha value '{vlm_provided_value}'. Setting value to None. UID: {target_persistent_uid}")
                                    effective_vlm_value = None
                    
                    if current_value_in_netlist_line is None or str(current_value_in_netlist_line).strip().lower() == 'none':
                        line_from_gen_netlist['value'] = effective_vlm_value
                    elif effective_vlm_value is None and \
                         prospective_component_type_from_vlm in ['V', 'I'] and \
                         (current_value_in_netlist_line is not None and str(current_value_in_netlist_line).strip().lower() != 'none'):
                        line_from_gen_netlist['value'] = None
                    
                    # Update class and component_type based on VLM
                    line_from_gen_netlist['class'] = vlm_class
                    line_from_gen_netlist['component_type'] = prospective_component_type_from_vlm
                    
                    if vlm_class == 'gnd':
                        line_from_gen_netlist['node_2'] = 0
                    
                    if self.debug:
                        print(f"Debug fix_netlist (Pass 1): Updated component UID {target_persistent_uid} (VisualID {visual_id_for_this_component}) with VLM data. New class: {vlm_class}, New type: {prospective_component_type_from_vlm}, New value: {line_from_gen_netlist['value']}")
                    break 
            
            if not found_vlm_match:
                if self.debug:
                    print(f"Debug fix_netlist (Pass 1): No VLM match found for component UID {target_persistent_uid} (VisualID {visual_id_for_this_component}). Original class/type preserved.")
                # Ensure 'class' and 'component_type' are consistent if no VLM update
                if 'class' not in line_from_gen_netlist:
                    line_from_gen_netlist['class'] = 'unknown'
                if 'component_type' not in line_from_gen_netlist:
                    line_from_gen_netlist['component_type'] = self.netlist_map.get(line_from_gen_netlist['class'], 'UN')


        # Sort netlist by visual_id to ensure stable numbering if visual_id exists.
        # Components without a visual_id (e.g., if not found in all_enumerated_bboxes) will be placed based on Python's default sort stability for None.
        # For robust sorting, convert visual_id to int if it's numeric, or handle None appropriately.
        def sort_key(item):
            vid = item.get('visual_id')
            if vid is None:
                return (float('inf'), item.get('persistent_uid')) # Place None visual_ids last, then by UID
            try:
                return (int(vid), item.get('persistent_uid'))
            except (ValueError, TypeError):
                return (float('inf'), item.get('persistent_uid')) # Fallback for non-integer visual_ids

        netlist.sort(key=sort_key)
        if self.debug:
            print("Debug fix_netlist: Netlist after sorting by visual_id:")
            for i, item_sorted in enumerate(netlist):
                 print(f"  Sorted item {i}: VisualID {item_sorted.get('visual_id')}, Class {item_sorted.get('class')}, Type {item_sorted.get('component_type')}, UID {item_sorted.get('persistent_uid')}")

        # Second Pass: Re-number all components sequentially based on their final types.
        final_component_counters = {comp_type: 1 for comp_type in set(self.netlist_map.values()) if comp_type}
        # Add a counter for 'UN' if it's not in netlist_map values or handle unknown types explicitly.
        if 'UN' not in final_component_counters:
            final_component_counters['UN'] = 1

        for line_item in netlist:
            component_type_final = line_item.get('component_type') # This is the VLM-derived type
            
            # Ensure component_type_final is valid and in counters
            if not component_type_final or component_type_final not in final_component_counters:
                if self.debug:
                    print(f"Debug fix_netlist (Pass 2): Encountered unexpected component_type '{component_type_final}' for UID {line_item.get('persistent_uid')}. Defaulting to 'UN'.")
                component_type_final = 'UN' # Default to UN if type is missing or invalid
                if 'UN' not in final_component_counters: # Ensure UN counter exists
                    final_component_counters['UN'] = 1
            
            # Assign new component_num. GND components usually don't get a number in SPICE,
            # but we can assign one internally if needed, or skip.
            # For now, we'll number all non-empty types.
            if component_type_final: # Only number if type is not empty string
                line_item['component_num'] = final_component_counters[component_type_final]
                final_component_counters[component_type_final] += 1
            else:
                # If component_type_final is an empty string (e.g. for 'junction'), it won't be numbered.
                # It might already lack a 'component_num' or have a placeholder.
                # We can explicitly remove/set to None if desired.
                line_item.pop('component_num', None) 
            
            if self.debug:
                print(f"Debug fix_netlist (Pass 2): Re-numbered component UID {line_item.get('persistent_uid')}. Final Type: {component_type_final}, New Num: {line_item.get('component_num')}")

    def stringify_line(self, netlist_line):
        # Skip ground components and components with empty type (like junctions)
        # from SPICE output.
        component_type = netlist_line.get('component_type')
        if netlist_line.get('class') == 'gnd' or not component_type:
            return ""
        
        # Ensure all necessary parts are present for stringification
        component_num = netlist_line.get('component_num')
        node_1 = netlist_line.get('node_1')
        node_2 = netlist_line.get('node_2')
        value = netlist_line.get('value', "None") # Default value to "None" string if not present

        if component_num is None or node_1 is None or node_2 is None:
            if self.debug:
                print(f"Debug stringify_line: Skipping line due to missing essential fields. Line: {netlist_line}")
            return "" # Cannot form a valid SPICE line

        return f"{component_type}{component_num} {node_1} {node_2} {value}"

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
            # SWAPPED: Default to node2_centroid (non-ground) as primary, node1_centroid (ground) as secondary
            return node2_centroid, node1_centroid

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
        model_to_use = "meta-llama/llama-4-maverick-17b-128e-instruct" 

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
                temperature=0.1, 
                max_tokens=1024, 
                top_p=0.98,         
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

        for bbox in bboxes:
            yolo_numeric_cls_id = bbox.get('_yolo_class_id_temp') 
            # Fallback to look up ID if _yolo_class_id_temp wasn't set (e.g. if reclassification failed to set it)
            if yolo_numeric_cls_id is None:
                string_class_name_for_id_lookup = bbox.get('class')
                for num_id, name_str in self.yolo_class_names_map.items():
                    if name_str == string_class_name_for_id_lookup:
                        yolo_numeric_cls_id = num_id
                        bbox['_yolo_class_id_temp'] = num_id # Also set it for consistency
                        break
            
            confidence = bbox.get('confidence', 0.0) # Keep confidence if needed for a threshold check later
            component_string_class_name = bbox.get('class')

            # MODIFIED CONDITION: Use class attributes for selecting components for LLaMA
            if yolo_numeric_cls_id is not None and \
               yolo_numeric_cls_id in self.llama_numeric_classes_of_interest and \
               component_string_class_name is not None and \
               component_string_class_name in self.llama_classes_of_interest_names: 
               # Add confidence check here if desired, e.g.: and confidence >= self.llama_confidence_threshold
                
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

    def reclassify_terminals_based_on_connectivity(self, image_rgb_original, bboxes_list_to_modify):
        """
        Performs a preliminary connectivity analysis to reclassify 'terminal' components.
        Modifies bboxes_list_to_modify in-place if a 'terminal' is connected to >= 2 nodes.
        
        Args:
            image_rgb_original (np.ndarray): The original image (RGB).
            bboxes_list_to_modify (List[Dict]): The list of bounding boxes from YOLO (e.g., bboxes_orig_coords_nms).
        """
        if self.debug:
            print("Starting preliminary reclassification of 'terminal' components...")

        # 1. Basic Segmentation and Contour Extraction (on a copy to avoid side effects)
        #    We use a simplified mask approach here, similar to what get_node_connections might do initially.
        #    This operates on the original, uncropped image.
        
        # Create a BGR version for OpenCV functions if needed, assuming image_rgb_original is RGB
        image_bgr_original = cv2.cvtColor(image_rgb_original, cv2.COLOR_RGB2BGR)
        
        # Get a basic segmented mask (e.g., black lines on white background)
        # Note: segment_circuit returns a binary inverted mask (lines are white 255, bg is 0)
        segmented_mask_for_prelim = self.segment_circuit(image_bgr_original) 

    
        prelim_wire_mask = segmented_mask_for_prelim.copy()
        components_to_ignore_for_emptying = ('crossover', 'junction', 'circuit', 'vss') # Similar to get_emptied_mask
        
        for bbox_prelim in bboxes_list_to_modify:
            if bbox_prelim.get('class') not in components_to_ignore_for_emptying:
                ymin, ymax = int(bbox_prelim['ymin']), int(bbox_prelim['ymax'])
                xmin, xmax = int(bbox_prelim['xmin']), int(bbox_prelim['xmax'])
                prelim_wire_mask[max(0, ymin):min(prelim_wire_mask.shape[0], ymax), 
                                 max(0, xmin):min(prelim_wire_mask.shape[1], xmax)] = 0
        
        if self.debug:
            self.show_image(prelim_wire_mask, "Prelim - Wire Mask for Terminal Reclassification")

        prelim_contours, _ = self.get_contours(prelim_wire_mask, area_threshold=0.0001) # Use a smaller threshold for prelim

        if self.debug:
            print(f"Prelim Reclass: Found {len(prelim_contours)} contours.")
            if not prelim_contours:
                print("Prelim Reclass: No contours found. Cannot reclassify terminals.")
                return # bboxes_list_to_modify remains unchanged

        # 2. Iterate through bboxes and check 'terminal' components
        voltage_dc_numeric_id = None
        for num_id, name_str in self.yolo.model.names.items():
            if name_str == 'voltage.dc':
                voltage_dc_numeric_id = num_id
                break
        
        if voltage_dc_numeric_id is None and self.debug:
            print("Prelim Reclass Warning: Could not find numeric ID for 'voltage.dc'. Reclassification might be incomplete.")

        for i, bbox_to_check in enumerate(bboxes_list_to_modify):
            if bbox_to_check.get('class') == 'terminal':
                connected_contour_ids = set()
                # Check connectivity to each contour. We need a pixel threshold for proximity.
                # The bboxes are in original image coordinates. Contours are also from original image mask.
                pixel_threshold_for_reclass = 10 # Adjustable

                for contour_item in prelim_contours:
                    contour_path = contour_item['contour'] # Array of points (x,y)
                    # Check if any point on the contour is near the bbox_to_check
                    for point_array in contour_path:
                        point_on_contour = tuple(point_array[0])
                        if self.is_point_near_bbox(point_on_contour, bbox_to_check, pixel_threshold_for_reclass):
                            connected_contour_ids.add(contour_item['id'])
                            break # This contour is connected, move to the next contour
                
                num_distinct_connections = len(connected_contour_ids)
                
                if self.debug:
                    print(f"Prelim Reclass: Terminal UID {bbox_to_check.get('persistent_uid')} connected to {num_distinct_connections} distinct contours.")

                if num_distinct_connections >= 2:
                    if self.debug:
                        print(f"Prelim Reclass: CHANGING class of UID {bbox_to_check.get('persistent_uid')} from 'terminal' to 'voltage.dc'.")
                    
                    original_yolo_class_name = bbox_to_check['class']
                    bbox_to_check['original_yolo_class_if_reclassified'] = original_yolo_class_name # Store original
                    bbox_to_check['class'] = 'voltage.dc' # New class
                    
                    if voltage_dc_numeric_id is not None:
                        bbox_to_check['_yolo_class_id_temp'] = voltage_dc_numeric_id
                    else:
                        if self.debug:
                             print(f"Prelim Reclass Warning: Numeric ID for 'voltage.dc' missing, _yolo_class_id_temp not updated for UID {bbox_to_check.get('persistent_uid')}.")
                    
                    # Mark that it was reclassified for later logic if needed
                    bbox_to_check['was_reclassified_from_terminal'] = True 
        if self.debug:
            print("Finished preliminary reclassification of 'terminal' components.")
        # bboxes_list_to_modify has been updated in-place.