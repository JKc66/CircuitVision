from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
import cv2
import numpy as np
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
                    model_cfg_path=str(sam2_config_path),
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
            self.last_sam2_output = None
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

            # Store for debugging visualization
            self.last_sam2_output = mask_binary_np.copy()
            
            # Create a colored version for better visualization
            colored_output = cv2.cvtColor(mask_binary_np, cv2.COLOR_GRAY2BGR)
            colored_output[:,:,0] = 0  # Set blue channel to 0
            colored_output[:,:,2] = 0  # Set red channel to 0
            # Now the mask will be visualized in green
            self.last_sam2_output = colored_output

            print("SAM 2 Segmentation successful.")
            return mask_binary_np

        except Exception as e:
            print(f"Error during SAM 2 segmentation: {e}")
            import traceback
            print(traceback.format_exc())
            self.last_sam2_output = None
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
                # Instead of fallback, raise an exception when SAM2 fails
                raise ValueError("SAM2 segmentation failed and no fallback is configured")
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
            # Instead of traditional segmentation, raise exception when SAM2 is disabled
            raise ValueError("SAM2 is disabled and no fallback is configured")
        
        if self.debug:
            self.show_image(emptied_mask, 'Emptied')
            
        enhanced = self.enhance_lines(emptied_mask)
        if self.debug:
            self.show_image(enhanced, 'Enhanced')
            
        contours, contour_image = self.get_contours(enhanced)
        if self.debug:
            self.show_image(contour_image, 'Contours')
            
        # Initialize nodes dictionary with all contours
        nodes = {i['id']:{'id': i['id'], 'components': [], 'contour': i['contour']} for i in contours}

        # Loop through components to find connections
        # Initialize rects_image for debug visualization of component-contour overlaps
        # This image will show the contours and then bounding boxes of components being checked.
        # If self.debug is true, it's shown at the end of this loop.
        # We make a fresh copy from contour_image as it's a good base.
        # Note: Previously rects_image was based on corners_image.
        rects_image = contour_image.copy()

        for i, bbox_comp in enumerate(resized_bboxes):
            if bbox_comp['class'] in self.non_components:
                continue
            
            # For debugging, can re-initialize rects_image per component if needed
            # rects_image = contour_image.copy() 
            xmin_comp, ymin_comp = int(bbox_comp['xmin']), int(bbox_comp['ymin'])
            xmax_comp, ymax_comp = int(bbox_comp['xmax']), int(bbox_comp['ymax'])
    
            for contour_item in contours: # Iterate over each contour (potential wire)
                # Bounding box of the current contour
                c_xmin_contour, c_ymin_contour, c_width_contour, c_height_contour = contour_item['rectangle']
                c_xmax_contour = c_xmin_contour + c_width_contour
                c_ymax_contour = c_ymin_contour + c_height_contour
                
                # Draw contour rectangle (for debugging visualization)
                cv2.rectangle(rects_image, (c_xmin_contour, c_ymin_contour), (c_xmax_contour, c_ymax_contour), color=(0, 255, 0), thickness=1) 
    
                # Broad phase: Check if component's bbox and contour's bbox overlap
                if xmax_comp < c_xmin_contour or xmin_comp > c_xmax_contour or ymax_comp < c_ymin_contour or ymin_comp > c_ymax_contour:
                    continue # No overlap between bounding boxes, so this contour cannot connect to this component
                
                # Narrow phase: Iterate through each point of the actual contour
                # contour_item['contour'] is a NumPy array of points (e.g., [[[x1,y1]], [[x2,y2]], ...])
                for point_array in contour_item['contour']:
                    point = tuple(point_array[0]) # Extract (x,y) from [[x,y]]
                    
                    if self.is_point_near_bbox(point, bbox_comp, pixel_threshold=6):
                        # A point on the contour is near the component's bbox
                        target_bbox_to_add = bboxes[i] if original_size else bbox_comp
                        
                        # Check if this component is already associated with this node to avoid duplicates.
                        # Construct a unique identifier for the component.
                        component_id_tuple = (
                            target_bbox_to_add['class'], 
                            target_bbox_to_add['xmin'], target_bbox_to_add['ymin'],
                            target_bbox_to_add['xmax'], target_bbox_to_add['ymax'],
                            target_bbox_to_add.get('id') # Include 'id' if present (e.g., from enumerate_components)
                        )

                        is_already_added = False
                        for existing_comp in nodes[contour_item['id']]['components']:
                            existing_comp_id_tuple = (
                                existing_comp['class'],
                                existing_comp['xmin'], existing_comp['ymin'],
                                existing_comp['xmax'], existing_comp['ymax'],
                                existing_comp.get('id')
                            )
                            if existing_comp_id_tuple == component_id_tuple:
                                is_already_added = True
                                break
                        
                        if not is_already_added:
                            nodes[contour_item['id']]['components'].append(deepcopy(target_bbox_to_add))
                            
                        # Connection found for this contour and this specific component.
                        # No need to check other points of this contour against this same component.
                        break 
        
        if self.debug:
            # This will show the rects_image for the last component processed, 
            # or it needs to be moved inside the component loop to show for each.
            # For now, keeping consistent with original placement.
            self.show_image(rects_image, "Nodes - Contour/BBox Overlaps")

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

        return new_nodes, emptied_mask, enhanced, contour_image, final_visualization

    


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