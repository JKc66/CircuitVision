# System Imports
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import sys, shutil
from os.path import join, realpath
import json
import xml.etree.ElementTree as ET
from xml import etree
from random import choice
import random
from copy import deepcopy
import cv2
from random import choice
import google.generativeai as genai
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
import colorsys
import random
import ast
import tempfile
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from typing import Union, Dict
import streamlit as st


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


def show_image(example):
    try:
        image = read_image(example)
    except:
        image = cv2.imread(example)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")  # Hide the axis
    plt.show()
        
def show_annotated(image, bboxes=None, name='Annotated'):
    image = image.copy()
    if bboxes == None:
        show_image(image)
        return
        
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
    show_image(image_rgb, name)
    
def find_best_position(xmin, ymin, xmax, ymax, text_width, text_height,
                          image_width, image_height, text_bboxes, step_size, max_distance):
    """Helper function to find the best position for text placement."""
    best_pos = None
    least_overlap = float('inf')

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

    def total_overlap_for_position(x, y, w, h, text_bboxes):
        total_overlap = 0
        for text_bbox in text_bboxes:
            overlap_area = calculate_overlap_area((x, y, x + w, y + h), text_bbox)
            total_overlap += overlap_area
        return total_overlap

    # Adaptive search around the bounding box
    for dx in range(-max_distance, max_distance + 1, step_size):
        for dy in range(-max_distance, max_distance + 1, step_size):
            pos_x = xmax + dx
            pos_y = ymax + dy

            # Calculate overlap even if position is out of bounds
            overlap = total_overlap_for_position(pos_x, pos_y, 
                                              text_width, text_height, 
                                              text_bboxes)

            # Track the position with the least overlap
            if overlap < least_overlap:
                least_overlap = overlap
                best_pos = (pos_x, pos_y)

    return best_pos
    
def show_image_with_annotations(example, class_=None): 
    image = read_image(example)
    
    if example['polygons']: 
        for bbox in example['polygons']:
            c = bbox['class']
            if class_:
                if c != class_:
                    continue
            b = bbox['bbox']
            xmin, ymin = int(b['xmin']), int(b['ymin'])
            xmax, ymax = int(b['xmax']), int(b['ymax'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)

            
            text_x = xmax
            text_y = ymax + 10  # Adjust the offset as needed
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 2
            color = (0, 0, 255)  # Red color
            cv2.putText(image, c, (text_x, text_y), font, font_scale, color, thickness)
            
    else:
        for bbox in example['bboxes']:
            c = bbox['class']
            if class_:
                if c != class_:
                    continue
            b = bbox
            xmin, ymin = int(b['xmin']), int(b['ymin'])
            xmax, ymax = int(b['xmax']), int(b['ymax'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
            
            try:
                r = bbox['rotation']
            except:
                r = None
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 1
            color = (0, 0, 255)  # Red color
            text_x = xmax
            text_y = ymax + 10  # Adjust the offset as needed
            if r != 'None' and r != None:
                cv2.putText(image, f"{c}, {r}deg", (text_x, text_y), font, font_scale, color, thickness)
            else:
                cv2.putText(image, f"{c}", (text_x, text_y), font, font_scale, color, thickness)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")  # Hide the axis
    plt.show()

def load_classes() -> dict:
    """Returns the List of Classes as Encoding Map"""

    with open("classes.json") as classes_file:
        return json.loads(classes_file.read())


def load_classes_ports() -> dict:
    """Reads Symbol Library from File"""

    with open("/kaggle/input/cghd1152/classes_ports.json") as json_file:
        return json.loads(json_file.read())
    
      
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
        
    genai.configure(api_key=api_key)
    
    # Convert the image file to PIL Image format if it’s a numpy array
    image_file = Image.fromarray(image_file) 
    
    #model = genai.GenerativeModel("gemini-1.5-flash-002")
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = ("Identify only the components and their values in this circuit schematic, the id of each component is in red. return the object as a python list of dictioaries."
              "If the values are just letters or don't exist for the component, the value should be None. If components are defined using complex values, write the complex/imaginary value."
              "Format the output as a list of dictionaries [{'class': 'voltage.dependent', 'value':'35*V_2', 'direction':'down', 'id': '1'}, "
              "{'class': 'voltage.ac', 'value':'22k:30', 'direction': 'None', 'id': '2'}], note how you always use the same ids of the components marked in red in the image, to identify the component. the closest red number to a component that component's id. nothing else. The value of the component should not specify the unit, only the suffix such as k or M or m or nothing, there shouldn't be a space between the number and the suffix. The classes included and their descriptions: " 
              + str(components_dict))
    # Generate content using the image and prompt
    result = model.generate_content([image_file, "\n\n", prompt], generation_config=genai.types.GenerationConfig(temperature=0.0))
    print(result.text)
    # Print the raw result to debug the response
    #print(result.text)  # Log the response for debugging
    formatted = result.text.strip('```python\n')
    formatted = formatted.strip('```json\n')
    formatted = formatted.strip('```')
    parsed_data = ast.literal_eval(formatted)  # Use json.loads instead of ast.literal_eval
#     for line in parsed_data:
#         line['value'] = parse_value(line['value'])
    return parsed_data

def load_properties() -> dict:
    """Loads the Properties RegEx File"""

    with open("/kaggle/input/cghd1152/properties.json") as json_file:
        return json.loads(json_file.read())


def _sample_info_from_path(path: str) -> tuple:
    """Extracts Sample Metadata from File Path"""

    drafter, _, file_name = os.path.normpath(path).split(os.sep)[-3:]
    circuit, drawing, picture = file_name.split("_")
    picture, suffix = picture.split(".")
    return drafter.split("_")[1], int(circuit[1:]), int(drawing[1:]), int(picture[1:]), suffix


def sample_name(sample: dict) -> str:
    """Returns the Name of a Sample"""

    return f"C{sample['circuit']}_D{sample['drawing']}_P{sample['picture']}"


def file_name(sample: dict) -> str:
    """return the Raw Image File Name of a Sample"""

    return f"{sample_name(sample)}.{sample['format']}"


def read_pascal_voc(path: str) -> dict:
    """Reads the Content of a Pascal VOC Annotation File"""

    root = ET.parse(path).getroot()
    circuit, drawing, picture = root.find("filename").text.split("_")
    drafter = int(os.path.normpath(path).split(os.sep)[-3].split("_")[1])

    return {"drafter": drafter,
            "circuit": int(circuit[1:]),
            "drawing": int(drawing[1:]),
            "picture": int(picture.split(".")[0][1:]),
            "format": picture.split(".")[-1],
            "width": int(root.find("size/width").text),
            "height": int(int(root.find("size/height").text)),
            "bboxes": [{"class": annotation.find("name").text,
                        "xmin": int(annotation.find("bndbox/xmin").text),
                        "xmax": int(annotation.find("bndbox/xmax").text),
                        "ymin": int(annotation.find("bndbox/ymin").text),
                        "ymax": int(annotation.find("bndbox/ymax").text),
                        "rotation": int(annotation.find("bndbox/rotation").text) if annotation.find("bndbox/rotation") is not None else None,
                        "text": annotation.find("text").text if annotation.find("text") is not None else None}
                       for annotation in root.findall('object')],
            "polygons": [], "points": []}


def read_labelme(path: str) -> dict:
    """Reads and Returns Geometric Objects from a LabelME JSON File"""

    with open(path) as json_file:
        json_data = json.load(json_file)

    drafter, circuit, drawing, picture, _ = _sample_info_from_path(path)
    suffix = json_data['imagePath'].split(".")[-1]

    return {'img_path': json_data['imagePath'].replace("\\", "/"), 'drafter': drafter, 'circuit': circuit,
            'drawing': drawing, 'picture': picture, 'format': suffix,
            'height': json_data['imageHeight'], 'width': json_data['imageWidth'], 'bboxes': [],
            'polygons': [{'class': shape['label'],
                          'bbox': {'xmin': min(point[0] for point in shape['points']),
                                   'ymin': min(point[1] for point in shape['points']),
                                   'xmax': max(point[0] for point in shape['points']),
                                   'ymax': max(point[1] for point in shape['points'])},
                          'points': shape['points'],
                          'rotation': shape.get('rotation', None),
                          'text': shape.get('text', None),
                          'group': shape.get('group_id', None)}
                         for shape in json_data['shapes']
                         if shape['shape_type'] == "polygon"],
            'points': [{'class': shape['label'], 'points': shape['points'][0],
                        'group': shape['group_id'] if 'group_id' in shape else None}
                       for shape in json_data['shapes']
                       if shape['shape_type'] == "point"]}


def read_dataset(drafter: int = None, circuit: int = None, segmentation=False, folder: str = None) -> list:
    """Reads all BB Annotation Files from Folder Structure
    This Method can be invoked from Anywhere, can be restricted to a specified drafter
    and can be use for both BB and Polygon Annotations. Alternative annotation sub-folder
    can be specified to read processed ground truth."""

    db_root = os.sep.join(realpath('/kaggle/input/cghd1152').split(os.sep))

    return sorted([(read_labelme if segmentation else read_pascal_voc)(join(root, file_name))
                   for root, _, files in os.walk(db_root)
                   for file_name in files
                   if (folder if folder else ("instances" if segmentation else "annotations")) in root and
                      (not circuit or f"C{circuit}_" in file_name) and
                      (drafter is None or f"drafter_{drafter}{os.sep}" in root)],
                  key=lambda sample: sample["circuit"]*100+sample["drawing"]*10+sample["picture"])


def read_image(sample: dict) -> np.ndarray:
    """Loads the Image Associated with a DB Sample"""

    db_root = os.sep.join(realpath('/kaggle/input/cghd1152').split(os.sep))

    return cv2.imread(join(db_root, f"drafter_{sample['drafter']}", "images", file_name(sample)))


def read_images(**kwargs) -> list:
    """Loads Images and BB Annotations and returns them as as List of Pairs"""

    return [(read_image(sample), sample) for sample in read_dataset(**kwargs)]

def count_instances(ds, classes_):
    instance_count = {name: 0 for name in classes_.keys()}
    class_sampled_ds = {name: [] for name in classes_.keys()}
    for sample in ds:
        if isinstance(sample, str) or sample_type(sample) == 'background':
            continue
        if sample_type(sample) == 'labelme':
            for bbox in sample['polygons']:
                c = bbox['class']
                if c == 'dirac':
                    continue
                instance_count[c] += 1 
                class_sampled_ds[c].append(sample)
        elif sample_type(sample) == 'voc':
            for bbox in sample['bboxes']:
                c = bbox['class']
                if c == 'dirac':
                    continue
                instance_count[c] += 1 
                class_sampled_ds[c].append(sample)

    return instance_count, class_sampled_ds



# ... (Import other necessary functions like read_dataset, 
#      load_classes, load_properties, count_instances, 
#      file_name, show_image_with_annotations, show_image) 

def calculate_iou(bbox1, bbox2):
    """Calculates IoU for bounding boxes in dictionary format."""
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


def get_path(d: dict, ds_loc='/kaggle/input/cghd1152', mask=False):
    if d["bboxes"]:
        name = file_name(d)
        old_path = os.path.join(ds_loc, f"drafter_{d['drafter']}", 'images', name)
    elif d['polygons']:
        name = d['img_path'].split("/")[-1]
        if mask:
            old_path = os.path.join(ds_loc, f"drafter_{d['drafter']}", 'segmentation', name)
        else:
            old_path = os.path.join(ds_loc, f"drafter_{d['drafter']}", 'images', name)
    elif d['background']:
        old_path = d['img_path']
    else:
        raise ValueError("Invalid sample dictionary format")
    return old_path


def sample_type(d: dict):
    if len(d.get("bboxes")) > 0:
        return 'voc'
    elif len(d.get("polygons")) > 0:
        return 'labelme'
    elif d.get('background') == True:
        return 'background'
    else:
        raise ValueError("Invalid sample dictionary format")


def get_bboxes(d: dict, non_component_classes=[], components_only=False):
    if d.get('bboxes'):
        t = 'bboxes'
        voc = True
    elif d.get('polygons'):
        t = 'polygons'
        voc = False
    elif d.get('background'):
        return []
    else:
        raise ValueError("Invalid sample dictionary format")

    bboxes = d[t]

    if not bboxes:
        raise ValueError("Empty bounding boxes list in sample")

    if voc:
        if not components_only:
            return [bbox for bbox in bboxes]
        else:
            return [bbox for bbox in bboxes if bbox['class'] not in non_component_classes]
    else:
        l = []
        for bbox in bboxes:
            b = bbox['bbox']
            b['class'] = bbox['class']
            if not components_only or (components_only and b['class'] not in non_component_classes):
                l.append(b)
        return l


def filter_dataset(combined_ds, class2category, non_component_classes, reducing: set, unknown: set, deleting: set):
    filtered_ds = deepcopy(combined_ds)
    for i, sample in enumerate(filtered_ds):
        if sample.get('bboxes') is not None:
            t = 'bboxes'
        elif sample.get('polygons') is not None:
            t = 'polygons'
        else:
            continue  # Skip samples without bounding boxes

        bboxes = sample[t]
        bboxes = [bbox for bbox in bboxes if bbox['class'] not in deleting]
        filtered_ds[i][t] = bboxes

        for j, bbox in enumerate(bboxes):
            if bbox['class'] in unknown:
                bbox['class'] = 'unknown'
            elif bbox['class'] in reducing:
                bbox['class'] = bbox['class'].split('.')[0]

    classes = set(class2category.keys()) - deleting - unknown - reducing
    class2category = {key: i for i, key in enumerate(classes)}
    instances_count, class_sorted_ds = count_instances(filtered_ds, class2category)
    return filtered_ds, classes, class2category, instances_count, class_sorted_ds


def establish_dirs(path='/kaggle/working/yolo_dataset/'):
    os.makedirs(os.path.join(path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'train', 'labels'), exist_ok=True)
    return True


def move_dataset_to(ds, filtered_dict, path='/kaggle/working/yolo_dataset/'):
    for sample in ds:
        sample_t = sample_type(sample)
        if sample_t == 'background':
            old_path = sample['img_path']
            file_name = old_path.split('/')[-1]
            new_path = os.path.join(path, 'train/images/', file_name)
            shutil.copyfile(old_path, new_path)
            with open(os.path.join(path, 'train/labels/', file_name.split('.')[0] + '.txt'), 'w') as file:
                pass
            continue

        if sample_t in ('voc', 'labelme'):
            old_path = get_path(sample)
            name = old_path.split('/')[-1]
        else:
            raise ValueError("Invalid sample type")

        new_path = os.path.join(path, 'train/images/', name)
        shutil.copyfile(old_path, new_path)
        h, w = cv2.imread(new_path).shape[:2]

        with open(os.path.join(path, 'train/labels/', name.split('.')[0] + '.txt'), 'w') as file:
            pass  # This clears the file's contents

        with open(os.path.join(path, 'train/labels/', name.split('.')[0] + '.txt'), 'a') as anno:
            for bbox in sample.get('bboxes', []) if sample_t == 'voc' else sample.get('polygons', []):
                if sample_t == 'labelme':
                    b = bbox['bbox']
                    xmin, ymin = int(b['xmin']), int(b['ymin'])
                    xmax, ymax = int(b['xmax']), int(b['ymax'])
                else:  # sample_t == 'voc'
                    xmin, ymin = int(bbox['xmin']), int(bbox['ymin'])
                    xmax, ymax = int(bbox['xmax']), int(bbox['ymax'])

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                width = (xmax - xmin) / w
                height = (ymax - ymin) / h

                class_index = filtered_dict.get(bbox['class'])
                if class_index is None:
                    print(f"Warning: Class '{bbox['class']}' not found in filtered_dict. Skipping...")
                    continue

                anno.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

                if width > 1 or height > 1 or x_center > 1 or y_center > 1:
                    print(f"Error: Invalid bounding box values - "
                          f"width: {width}, height: {height}, x_center: {x_center}, y_center: {y_center}")
                    print(f"Sample width: {w}, height: {h}")
                    print(f"Bounding box: {bbox}")
                    show_image_with_annotations(sample)
                    raise ValueError("Invalid bounding box values detected")


def get_mask(sample, ds_loc='/kaggle/input/cghd1152'):
    if sample_type(sample) == 'labelme':
        return cv2.imread(get_path(sample, ds_loc=ds_loc, mask=True), cv2.IMREAD_GRAYSCALE)
    else:
        return None


def get_filtered_mask(sample, non_component_classes):
    mask = get_mask(sample)
    if mask is None:
        return None

    image_copy = mask.copy()  # Create a copy to avoid modifying the original mask

    for bbox in get_bboxes(sample, non_component_classes):
        if bbox['class'] in ('text', 'explanatory'):
            image_copy[int(bbox['ymin']):int(bbox['ymax']), int(bbox['xmin']):int(bbox['xmax'])] = 255
    return image_copy


def get_emptied_mask(sample, non_component_classes):
    mask = get_mask(sample)
    if mask is None:
        return None

    image_copy = mask.copy()  # Create a copy to avoid modifying the original mask

    for bbox in get_bboxes(sample, non_component_classes):
        if bbox['class'] not in ('crossover', 'junction', 'terminal'):
            image_copy[int(bbox['ymin']):int(bbox['ymax']), int(bbox['xmin']):int(bbox['xmax'])] = 255
    return image_copy

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