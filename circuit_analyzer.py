from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO
from utilities import *

class CircuitAnalyzer():
    def __init__(self, yolo_path='/kaggle/input/circuit/best_large.pt', debug=False):
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
        
        self.class2num = self.classes
        self.num2class = {value: key for key, value in self.classes.items()}
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
            'voltage.dependent': 'E', # Added dependent voltage source
            'current.dc': 'I',        # Added DC current source
            'current.dependent': 'G', # Added dependent current source
            'vss': 'GND',
            'gnd': '0',
            'switch': 'S',
            'integrated_circuit': 'X',
            'integrated_circuit.voltage_regulator': 'X',
            'operational_amplifier': 'X',
            'thyristor': 'Q',
            'transformer': 'T',
            'varistor': 'RV',  # Note: potential conflict with voltage sources. Consider using 'RV'
            'terminal': 'N',
            'junction': '',
            'crossover': '',
            'explanatory': '',
            'text': '',
            'unknown': 'UN',
        }
        
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

    def enumerate_components_from_ds(self, example, excluded_labels=None):
        # Read the image using a helper function (customize it as per your dataset structure)
        image = read_image(example)
        
        if excluded_labels == None:
            excluded_labels = self.non_components
            
        # Get the image dimensions
        image_height, image_width = image.shape[:2]

        # Calculate a relative font scale based on the image height
        font_scale = max(0.5, image_height / 800.0)  # You can adjust the denominator to scale as desired
        thickness = int(max(1, image_height / 400.0))  # Adjust the thickness relative to the image size

        bbox_counter = 0  # Initialize counter to number each bbox

        # Check for 'polygons' first
        if 'polygons' in example and example['polygons']:
            for bbox in example['polygons']:
                c = bbox['class']
                if excluded_labels and c in excluded_labels:
                    continue  # Skip if the class is in the excluded set

                b = bbox['bbox']
                xmin, ymin = int(b['xmin']), int(b['ymin'])
                xmax, ymax = int(b['xmax']), int(b['ymax'])

                # Draw the bounding box in red
                #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=thickness)

                # Increment counter for each bbox and assign a number
                bbox_counter += 1

                # Draw the class name and number
                text_x = xmax + int(10 * font_scale)
                text_y = ymax + int(10 * font_scale)  # Adjust the offset based on the font scale
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255)  # Red color for the text

                # Draw the class name with the number
                cv2.putText(image, f"{bbox_counter}", (text_x, text_y), font, font_scale, color, thickness)

        # Check for 'bboxes' if 'polygons' is not found
        elif 'bboxes' in example and example['bboxes']:
            for bbox in example['bboxes']:
                c = bbox['class']
                if excluded_labels and c in excluded_labels:
                    continue  # Skip if the class is in the excluded set

                b = bbox
                xmin, ymin = int(b['xmin']), int(b['ymin'])
                xmax, ymax = int(b['xmax']), int(b['ymax'])

                # Draw the bounding box in red
                #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=thickness)

                # Increment counter for each bbox and assign a number
                bbox_counter += 1

                # Draw the class name, number, and optional rotation if available
                text_x = xmax + int(10 * font_scale)
                text_y = ymax + int(10 * font_scale)  # Adjust the offset as needed
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255)  # Red color for the text

                # Add rotation information if available
                try:
                    r = bbox['rotation']
                except KeyError:
                    r = None

                if r != 'None' and r is not None:
                    cv2.putText(image, f"{bbox_counter}", (text_x, text_y), font, font_scale, color, thickness)
                else:
                    cv2.putText(image, f"{bbox_counter}", (text_x, text_y), font, font_scale, color, thickness)

        # Convert image from BGR to RGB for displaying with Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        if self.debug:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            plt.axis("off")  # Hide axis
            plt.show()


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
    
    def show_annotated(self, image, bboxes=None, name='Annotated'):
        image = image.copy()
        if bboxes == None:
            bboxes = self.bboxes(image)
        for bbox in bboxes:
            c = bbox['class']
            b = bbox
            xmin, ymin = int(b['xmin']), int(b['ymin'])
            xmax, ymax = int(b['xmax']), int(b['ymax'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            thickness = 1
            color = (0, 0, 255)  # Red color
            text_x = xmax
            text_y = ymax + 10  # Adjust the offset as needed
            cv2.putText(image, f"{c}", (text_x, text_y), font, font_scale, color, thickness)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        self.show_image(image_rgb, name)
    
    def get_annotated(self, image, bboxes=None, name='Annotated'):
        image = image.copy()
        if bboxes == None:
            bboxes = self.bboxes(image)
        for bbox in bboxes:
            c = bbox['class']
            b = bbox
            xmin, ymin = int(b['xmin']), int(b['ymin'])
            xmax, ymax = int(b['xmax']), int(b['ymax'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            thickness = 1
            color = (0, 0, 255)  # Red color
            text_x = xmax
            text_y = ymax + 10  # Adjust the offset as needed
            cv2.putText(image, f"{c}", (text_x, text_y), font, font_scale, color, thickness)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        return image
    
    def get_node_connections(self, image, bboxes, original_size=False):
        original = image.copy()
        image = image.copy()
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
                
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates)
            dbscan = DBSCAN(eps=0.07, min_samples=3)  # Experiment with these parameters!
            labels = dbscan.fit_predict(coordinates_scaled)
            unique_labels = set(labels)
            centers = []
            for label in unique_labels:
                if label != -1:  # Ignore noise points (-1)
                    cluster_points = coordinates[labels == label]
                    center = np.mean(cluster_points, axis=0)
                    centers.append(center)
            centers = np.array(centers)
            
            # Apply k-means clustering 
            #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
            #compactness, labels, centers = cv2.kmeans(coordinates, K=30, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
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

    
    def get_valid_texts(self, bboxes):
        texts = [bbox for bbox in bboxes if bbox['class'] == 'text']
        filtered = non_max_suppression_by_area(texts, iou_threshold=0.5)
        return filtered

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
    
    def process_image_to_netlist(self, image_path):
        image = cv2.imread(image_path)
            
        # Show and process the image to get bounding boxes and netlist
        if self.debug:
            self.show_image(image)
        bboxes = self.bboxes(image)
        bboxes = non_max_suppression_by_area(bboxes, iou_threshold=0.6)
        if self.debug:
            self.show_annotated(image, bboxes)

        # Generate nodes and netlist
        nodes = self.get_node_connections(image, bboxes, original_size=True)
        netlist = self.generate_netlist_from_nodes(nodes)
        if self.debug:
            for n in nodes:
                print(f"Node {n['id']}:", [i['class'] for i in n['components']])
            
            #print(netlist)
        # Get component information and fix netlist
        #enum_img, _ = self.enumerate_components(image, netlist)
        #gemini_info = gemini_labels(enum_img)
        #self.fix_netlist(netlist, gemini_info)
        return netlist, nodes

    def netlsit_aligns_with_project(self, netlist):
        for i in netlist:
            if i['class'] not in self.project_classes and i['class'].split('.')[0] not in self.project_classes:
                return False
        return True

    def image_aligns_with_project(self, image):
        bboxes = self.bboxes(image)
        for i in bboxes:
            if i['class'] in self.non_components:
                continue
            if i['class'] not in self.project_classes and i['class'].split('.')[0] not in self.project_classes and i['class'] != 'unknown':
                return False
        return True

    def validate_circuit(self, nodes):
        """
        Validates if a circuit configuration is valid based on the following rules:
        1. Must have at least one source component
        2. Must have at least one ground component
        3. Each node must have at least 2 components (except possibly terminal nodes)
        4. Circuit must be connected (no isolated components/nodes)
        5. Components must have valid connections
        
        Args:
            nodes: List of dictionaries containing node information with format:
                [{'id': int, 'components': [{...}], 'contour': array}, ...]
        
        Returns:
            tuple: (bool, str) - (is_valid, error_message)
        """
        if not nodes:
            return False, "No nodes found in the circuit"
        
        # Check for source and ground
        all_components = []
        for node in nodes:
            all_components.extend(node['components'])
        
        has_source = any(comp['class'] in self.source_components for comp in all_components)
        
        if not has_source:
            return False, "Circuit missing power source"
            
        def find_connected_nodes(start_node, nodes_list):
            connected = set([start_node['id']])
            stack = [start_node]
            
            while stack:
                current_node = stack.pop()
                # Create a unique identifier for each component based on its bbox coordinates
                current_components = set(
                    f"{comp.get('xmin', '')},{comp.get('ymin', '')},{comp.get('xmax', '')},{comp.get('ymax', '')}"
                    for comp in current_node['components']
                )
                
                for node in nodes_list:
                    if node['id'] not in connected:
                        node_components = set(
                            f"{comp.get('xmin', '')},{comp.get('ymin', '')},{comp.get('xmax', '')},{comp.get('ymax', '')}"
                            for comp in node['components']
                        )
                        # If there's any shared component, nodes are connected
                        if current_components & node_components:
                            connected.add(node['id'])
                            stack.append(node)
            
            return connected
        
        # Start from node 0 (which should be the main power node based on the original code)
        connected_nodes = find_connected_nodes(nodes[0], nodes)
        
        if len(connected_nodes) != len(nodes):
            return False, "Circuit contains disconnected sections"
        
        # Check component counts per node
        for node in nodes:
            component_count = len(node['components'])
            if component_count < 2:
                return False, f"Node {node['id']} has insufficient connections ({component_count})"
            
            # Check for invalid component combinations
            component_classes = [comp['class'] for comp in node['components']]
            
            # Rule: A node shouldn't connect multiple sources directly
            if component_classes.count('source') > 1:
                return False, f"Node {node['id']} has multiple sources connected"
            
            # Rule: A node shouldn't connect multiple grounds directly
            if component_classes.count('ground') > 1:
                return False, f"Node {node['id']} has multiple grounds connected"
        
        # Check for short circuits (direct source to ground connection)
        for node in nodes:
            component_classes = set(comp['class'] for comp in node['components'])
            if 'source' in component_classes and 'ground' in component_classes:
                return False, f"Short circuit detected at node {node['id']}"
        
        # If all checks pass, the circuit is valid
        return True, "Circuit configuration is valid"

            
    def stringify_line(self, netlist_line):
        return f"{netlist_line['component_type']}{netlist_line['component_num']} {netlist_line['node_1']} {netlist_line['node_2']} {netlist_line['value']}"