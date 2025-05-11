# System Imports
import os
from dotenv import load_dotenv
import json
import cv2
from google import genai
from google.genai import types
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import ast
import re
from typing import Union, Dict, List
import streamlit as st
import openai
import base64
import io
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

components_dict = {
    'gnd': 'Ground: A reference point in an electrical circuit. Its value is None.',
    'voltage.ac': 'AC Voltage source. If its value is written in phasor, format it as magnitude:phase.',
    'voltage.dc': 'DC Voltage source. Its polarity is important for circuit analysis.',
    'voltage.battery': 'Battery Voltage source. Its polarity is important for circuit analysis.',
    'resistor': 'Resistor: A passive component.',
    'voltage.dependent': 'Voltage-Dependent Source: A voltage source whose output voltage depends on another voltage or current in the circuit. Its polarity is important.',
    'current.dc': 'DC Current: Direct current, where the current flows in one direction consistently. Its direction of flow is important.',
    'current.dependent': 'Current-Dependent Source: A current source whose output current depends on another current or voltage in the circuit. Its direction of flow is important.',
    'capacitor': 'Capacitor: A passive component.',
    'inductor': 'Inductor: A passive component.',
    'diode': 'Diode: A semiconductor device that primarily conducts current in one direction. Its orientation is important.'
}

# Gemini prompt for component analysis
PROMPT = """
You are an expert electrical engineering assistant. Your task is to analyze an image of a circuit schematic.
In the image, electrical components are marked with red ID numbers.
Your goal is to identify these components and their values.

Output your findings as a Python list of dictionaries. Each dictionary in the list represents one component.
Strictly adhere to the following format for each dictionary:
[
  {
    "id": "string_id_from_image",
    "class": "component_class_name",
    "value": "component_value_string_or_null"
  }
  // ... more components can follow
]

Example of a single component entry:
{
    "id": "1",
    "class": "voltage.ac",
    "value": "10:30"
}
{
    "id": "2",
    "class": "resistor",
    "value": "10k"
}


Key Instructions for each field in the dictionary:

1.  **`id` (String):**
    *   This MUST be the red number shown next to the component in the image.
    *   The value for 'id' MUST be a STRING (e.g., "1", "12", "27").

2.  **`class` (String):**
    *   Use ONLY the class names provided as keys in the 'Component Classes and Descriptions' section below (e.g., 'resistor', 'voltage.ac').
    *   Do not invent new class names.

3.  **`value` (String or null/None):**
    *   If a numerical value is present:
        *   Represent it as a STRING.
        *   Include metric prefixes directly attached to the number if present (e.g., "10k", "2.2M", "100m", "0.5u", "22n", "47p"). NO SPACE between number and prefix.
        *   Do NOT include the base unit (like Ω, F, V, A). Just the number and prefix.
        *   For AC voltage sources (`voltage.ac`), if a phasor is given, format the value string as "magnitude:angle_in_degrees" (e.g., "120:30" for 120V at 30 degrees, "10:0").
        *   For complex impedance values (e.g., for capacitors or inductors if given in ohms), use the format "R+jX" or "R-jX" as a string (e.g., "5+j3.14", "100-j50").
    *   If the value is a variable name or an expression (e.g., "V_in", "R_load", "X1", "35*V_2"), use that variable name or expression as a STRING.
    *   If no value is explicitly written next to the component on the schematic, or if it's unclear (e.g., a question mark "?"), the value MUST be `null` (if generating JSON) or `None` (if generating a Python literal string).

General Instructions:

*   Identify ONLY the components that have a clear red ID number next to them.
*   The 'id' in your output dictionary MUST correspond to this red number.
*   If a component in the image is ambiguous, its ID is unclear, or it cannot be confidently classified using the provided list, DO NOT include it in the output list.
*   Ensure the entire output is a valid Python list of dictionaries string, parsable by `ast.literal_eval`, or a valid JSON array of objects.

Component Classes and Descriptions:
""" + str(components_dict)


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
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        
    client = genai.Client(api_key=api_key)
    
    # Convert the image file to PIL Image format if it's a numpy array
    image_file = Image.fromarray(image_file) 
    
    # MODEL = "gemini-2.5-flash-preview-04-17"
    MODEL = "gemini-2.5-pro-exp-03-25"
    
    # Generate content using the image and prompt
    response = client.models.generate_content(
        model=MODEL,
        contents=[image_file, "\n", PROMPT],
        config=types.GenerateContentConfig(
            temperature=0
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

def gemini_labels_openrouter(image_file):
    """
    Uploads an image of a circuit schematic to OpenRouter (using OpenAI SDK)
    to analyze components with a Gemini model, retrieves the components and
    their values, and returns them in a structured format.

    Args:
        image_file (np.ndarray): Image array of the circuit schematic.

    Returns:
        dict: A dictionary with component names as keys and a list of their
              values as entries.
    """
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file.")
        
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        default_headers={
            "HTTP-Referer": "https://app.jawadk.me/circuits/",
            "X-Title": "Circuit Analyzer"
        }
    )
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_file)

    # Convert PIL Image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Using OpenRouter's model
    # OPENROUTER_MODEL = "google/gemini-2.5-pro-preview"
    OPENROUTER_MODEL = "google/gemini-2.5-flash-preview"

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
        )
        
        if not response.choices or len(response.choices) == 0:
            raise ValueError("OpenRouter API response did not contain any choices.")
        
        response_text = response.choices[0].message.content
        print(response_text)
        
        # Clean up the response text
        formatted = response_text.strip('```python\n')
        formatted = formatted.strip('```json\\n')
        formatted = formatted.strip('```')
        
        try:
            # Primary attempt: ast.literal_eval for Python-style list of dicts.
            # Prompt asks for "a python list of dictioaries".
            # Replace 'null' with 'None' in case the LLM uses 'null' (valid in JSON)
            # within a structure that is otherwise a Python literal (e.g. single quotes for keys).
            # This makes ast.literal_eval more robust to LLM variations.
            prepared_for_ast = formatted.replace('null', 'None')
            parsed_data = ast.literal_eval(prepared_for_ast)
            # If successful, can add print("Successfully parsed with ast.literal_eval.") for debugging
            return parsed_data
        except (SyntaxError, ValueError) as e_ast:
            # Log that the primary attempt failed and we're trying the fallback.
            print(f"Primary parsing (ast.literal_eval) failed: {e_ast}. Trying fallback (json.loads)...")
            # It can be helpful to see the string that failed ast.literal_eval
            # print(f"String passed to ast.literal_eval (after .replace('null','None')): {prepared_for_ast}")
            try:
                # Fallback: Try json.loads.
                # This would work if the LLM returned a strictly valid JSON string.
                # No .replace() needed here for 'None' to 'null', as 'formatted' is the original
                # string from LLM (after stripping ```). If it's JSON, it should use 'null'.
                parsed_data = json.loads(formatted)
                # If successful, can add print("Successfully parsed with json.loads as fallback.") for debugging
                return parsed_data
            except json.JSONDecodeError as e_json:
                # Both parsing methods failed. Log detailed error and the problematic string.
                print(f"Fallback parsing (json.loads) also failed: {e_json}")
                print(f"Original formatted string that failed both parsers: {formatted}")
                # Re-raise a comprehensive error.
                raise ValueError(f"Failed to parse OpenRouter response. ast.literal_eval error: {e_ast}, json.loads error: {e_json}. Formatted response: {formatted}")
    except openai.APIError as e:
        print(f"OpenRouter API Error with model {OPENROUTER_MODEL}: {e}")
        raise ValueError(f"OpenRouter API request failed: {str(e)}")

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

def create_annotated_image(image: np.ndarray, bboxes: List[Dict]) -> np.ndarray:
    """
    Creates an annotated image with bounding boxes and labels.

    Args:
        image (np.ndarray): The original image.
        bboxes (List[Dict]): A list of bounding box dictionaries.
                             Each dictionary should have 'xmin', 'ymin', 'xmax', 'ymax', 
                             'class', and 'confidence'.

    Returns:
        np.ndarray: The annotated image.
    """
    annotated_image = image.copy()
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
    return annotated_image

def calculate_component_stats(bboxes: List[Dict]) -> Dict:
    """
    Calculates statistics for detected components.

    Args:
        bboxes (List[Dict]): A list of bounding box dictionaries.
                             Each dictionary should have 'class' and 'confidence'.

    Returns:
        Dict: A dictionary where keys are component class names and values are 
              dictionaries containing 'count' and 'total_conf'.
    """
    component_stats = {}
    for bbox in bboxes:
        name = bbox['class']
        conf = bbox['confidence']
        if name not in component_stats:
            component_stats[name] = {'count': 0, 'total_conf': 0}
        component_stats[name]['count'] += 1
        component_stats[name]['total_conf'] += conf
    return component_stats

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
    

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to format EXIF data for better display
def format_value(value):
    """Format values for display, handling binary data appropriately"""
    if isinstance(value, bytes):
        return f"[Binary data, {len(value)} bytes]"
    elif isinstance(value, str):
        cleaned = ''.join(c for c in value if c.isprintable())
        return cleaned if cleaned else "[Empty string]"
    return value

def format_exif_data(image_path):
    """
    Extracts and formats key EXIF data from an image for display.
    Returns a dictionary of important EXIF tags or None if no data is found.
    """
    try:
        img = Image.open(image_path)
        
        # Define important tags we want to show
        important_tags = {
            'Software',
            'Orientation'
        }
        
        exif_data = {}
        try:
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in important_tags:
                        exif_data[tag] = format_value(value)
        except Exception as e:
            logger.warning(f"Error getting EXIF with _getexif(): {e}")
            
        return exif_data if exif_data else None
        
    except Exception as e:
        logger.error(f"Error formatting EXIF data: {str(e)}")
        return None

def safe_to_complex(value):
    """
    Safely convert PySpice values to complex numbers for AC analysis.
    
    Args:
        value: A value that could be a PySpice.Unit.Unit.UnitValue, int, float, or complex
        
    Returns:
        complex: A Python complex number
    """
    try:
        # If it's already a complex number
        if isinstance(value, complex):
            print(f"DEBUG safe_to_complex: Input is already complex: {value}")
            return value
            
        # If it's a PySpice UnitValue, extract the numerical value
        if hasattr(value, 'value'):
            print(f"DEBUG safe_to_complex: Input is UnitValue. value.value is: {value.value}, type: {type(value.value)}")
            return complex(value.value)
            
        # If it's a simple number type
        if isinstance(value, (int, float)):
            print(f"DEBUG safe_to_complex: Input is int/float: {value}")
            return complex(value)
            
        # Try generic conversion as last resort
        print(f"DEBUG safe_to_complex: Trying generic complex() conversion for: {value}, type: {type(value)}")
        return complex(value)
        
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Warning: Could not convert {type(value)} to complex: {value}, Error: {e}")
        # Return a default value to prevent further errors
        return complex(0)




# Helper function to parse AC value strings from VLM
def _parse_vlm_ac_string(raw_value_str):
    if not isinstance(raw_value_str, str):
        return None

    # Pattern to capture: AC <mag_val> <mag_unit> <freq_val> <freq_unit> <phase_val> <phase_unit>
    # Example: "AC 5V 1kHz 0deg" -> mag=5, phase=0
    # Example: "AC 10.5mA 50.2Hz -45.5deg" -> mag=10.5, phase=-45.5
    pattern_long = re.compile(
        r"AC\s*"                                      # "AC "
        r"([+-]?\d*\.?\d+)\s*[a-zA-ZμmkKVAMWΩ°]*\s*"     # Magnitude value and optional unit
        r"(?:[+-]?\d*\.?\d+)\s*[a-zA-ZμmkKVAMWΩHz°]*\s*" # Frequency value and optional unit (non-capturing)
        r"([+-]?\d*\.?\d+)\s*[a-zA-ZμmkKVAMWΩ°deg]*",   # Phase value and optional unit
        re.IGNORECASE
    )
    match_long = pattern_long.match(raw_value_str.strip())
    if match_long:
        try:
            mag_str = match_long.group(1)
            phase_str = match_long.group(2) # Phase is the second captured group
            mag = float(mag_str)
            phase = float(phase_str)
            return {'dc_offset': 0, 'mag': mag, 'phase': phase}
        except (IndexError, ValueError):
            pass # Will try other patterns

    # Fallback for "AC <mag_val><unit> <phase_val><unit>" if freq is missing
    # Example: "AC 5V 0deg"
    pattern_short = re.compile(
        r"AC\s*"
        r"([+-]?\d*\.?\d+)\s*[a-zA-ZμmkKVAMWΩ°]*\s*"  # Magnitude value and optional unit
        r"([+-]?\d*\.?\d+)\s*[a-zA-ZμmkKVAMWΩ°deg]*", # Phase value and optional unit
        re.IGNORECASE
    )
    match_short = pattern_short.match(raw_value_str.strip())
    if match_short:
        try:
            mag_str = match_short.group(1)
            phase_str = match_short.group(2)
            mag = float(mag_str)
            phase = float(phase_str)
            return {'dc_offset': 0, 'mag': mag, 'phase': phase}
        except (IndexError, ValueError):
            pass # Will try other patterns
    
    # NEW: Try "<mag>:<phase>" format, e.g., "4:-45" or "1:45"
    pattern_mag_phase = re.compile(
        r"\s*([+-]?\d*\.?\d+)\s*:\s*([+-]?\d*\.?\d+)\s*"
    )
    match_mag_phase = pattern_mag_phase.fullmatch(raw_value_str.strip())
    if match_mag_phase:
        try:
            mag = float(match_mag_phase.group(1))
            phase = float(match_mag_phase.group(2))
            return {'dc_offset': 0, 'mag': mag, 'phase': phase}
        except (IndexError, ValueError):
            pass # Should not happen if regex fullmatch succeeds

    return None # If no patterns match
