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
    
    MODEL = "gemini-2.5-flash-preview-04-17"
    # MODEL = "gemini-2.5-pro-exp-03-25"
    
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
    
    # Clean up the response text
    formatted = response.text.strip('```python\n')
    formatted = formatted.strip('```json\n')
    formatted = formatted.strip('```')
    
    try:
        import json
        # Parse using json.loads() instead of ast.literal_eval() to handle null values properly
        parsed_data = json.loads(formatted)
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {formatted}")
        # Fallback: Try to convert null to None for ast.literal_eval
        try:
            formatted = formatted.replace('null', 'None')
            parsed_data = ast.literal_eval(formatted)
            return parsed_data
        except Exception as e2:
            print(f"Fallback parsing also failed: {e2}")
            raise ValueError(f"Failed to parse Gemini response: {e2}. Original response: {formatted}")
#     for line in parsed_data:
#         line['value'] = parse_value(line['value'])

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