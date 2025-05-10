import torch
import numpy as np
import cv2
import shutil
import streamlit as st
import logging
import os
import time
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS
from src.utils import (summarize_components,
                        gemini_labels,
                        gemini_labels_openrouter,
                        non_max_suppression_by_confidence,
                        create_annotated_image,
                        calculate_component_stats
                        )
from src.circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Spice.Parser import SpiceParser
from PySpice.Unit import *
from PySpice.Unit import u_Hz
import re
import matplotlib.pyplot as plt


# Configure logging
# Get log level from environment or default to DEBUG
log_level = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(name)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("circuit_analyzer")
logger.info(f"Initializing Circuit Analyzer with log level: {log_level}")

# Reduce verbosity of httpcore and openai loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# Suppress PyTorch model parameter logs
logging.getLogger("torch.nn.modules").setLevel(logging.WARNING)
logging.getLogger("torch.nn.parameter").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

torch.classes.__path__ = []

# Set page config and custom styles
st.set_page_config(
    page_title="CircuitVision",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "AI-powered circuit diagram analyzer that detects components, analyzes connections, and generates SPICE netlists instantly."
    }
)

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Function to auto-rotate image based on EXIF orientation tag
def auto_rotate_image(image_path):
    """
    Opens an image, checks for EXIF orientation tag, and rotates the image accordingly.
    Returns the rotated PIL Image object, or the original if no rotation is needed/possible.
    """
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        
        # If there's no EXIF data or no orientation tag, return the original image
        if not exif or 0x0112 not in exif:  # 0x0112 is the orientation tag ID
            return image
        
        orientation = exif[0x0112]
        logger.info(f"Auto-rotating image with orientation tag: {orientation}")
        
        rotated_image = ImageOps.exif_transpose(image)
        return rotated_image
    except Exception as e:
        logger.error(f"Error auto-rotating image: {str(e)}")
        return Image.open(image_path)

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

# Load custom CSS
load_css("static/css/main.css")

# Set up base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static/uploads')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models/YOLO/best_large_model_yolo.pt')
# Set SAM2 paths as absolute path objects
SAM2_CONFIG_PATH_OBJ = os.path.join(BASE_DIR, 'models/configs/sam2.1_hiera_l.yaml')
SAM2_BASE_CHECKPOINT_PATH_OBJ = os.path.join(BASE_DIR, 'models/SAM2/sam2.1_hiera_large.pt')
SAM2_FINETUNED_CHECKPOINT_PATH_OBJ = os.path.join(BASE_DIR, 'models/SAM2/best_miou_model_SAM_latest.pth')
# Convert to absolute paths for SAM2
SAM2_CONFIG_PATH = "/" + os.path.abspath(SAM2_CONFIG_PATH_OBJ)  # Config path needs leading slash for Hydra
SAM2_BASE_CHECKPOINT_PATH = os.path.abspath(SAM2_BASE_CHECKPOINT_PATH_OBJ)
SAM2_FINETUNED_CHECKPOINT_PATH = os.path.abspath(SAM2_FINETUNED_CHECKPOINT_PATH_OBJ)

# logger.info(f"Base directory: {BASE_DIR}")
# logger.info(f"YOLO model path: {YOLO_MODEL_PATH}")
# logger.info(f"SAM2 config path: {SAM2_CONFIG_PATH}")
# logger.info(f"SAM2 base checkpoint path: {SAM2_BASE_CHECKPOINT_PATH}")
# logger.info(f"SAM2 finetuned checkpoint path: {SAM2_FINETUNED_CHECKPOINT_PATH}")


# Create necessary directories
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
sam2_dir = os.path.join(BASE_DIR, 'models/SAM2')
if not os.path.exists(sam2_dir):
    os.makedirs(sam2_dir, exist_ok=True)  # Ensure models/SAM2 exists
logger.info(f"Created necessary directories: {UPLOAD_DIR}")

# Check if SAM2 model files exist
if not os.path.exists(SAM2_CONFIG_PATH_OBJ) or not os.path.exists(SAM2_BASE_CHECKPOINT_PATH_OBJ) or not os.path.exists(SAM2_FINETUNED_CHECKPOINT_PATH_OBJ):
    missing_files = []
    if not os.path.exists(SAM2_CONFIG_PATH_OBJ):
        missing_files.append(f"Config: {SAM2_CONFIG_PATH}")
    if not os.path.exists(SAM2_BASE_CHECKPOINT_PATH_OBJ):
        missing_files.append(f"Base Checkpoint: {SAM2_BASE_CHECKPOINT_PATH}")
    if not os.path.exists(SAM2_FINETUNED_CHECKPOINT_PATH_OBJ):
        missing_files.append(f"Fine-tuned Checkpoint: {SAM2_FINETUNED_CHECKPOINT_PATH}")
    
    logger.warning("One or more SAM2 model files not found:\n" + "\n".join(missing_files) + "\nSAM2 features will be disabled.")
    st.warning("One or more SAM2 model files not found:\n" + "\n".join(missing_files) + "\nSAM2 features will be disabled.")
    use_sam2_feature = False
else:
    logger.info("All SAM2 model files found and will be used")
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
            debug=True
        )
        return analyzer
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        import traceback
        trace_msg = f"Traceback: {traceback.format_exc()}"
        logger.error(trace_msg)
        st.error(error_msg)
        st.error(trace_msg)
        return None

# Initialize the circuit analyzer
analyzer = load_circuit_analyzer()

# Check if analyzer loaded successfully before proceeding
if analyzer is None:
    logger.error("Circuit analyzer failed to initialize. Stopping application.")
    st.stop()  # Stop the app if model loading failed
else:
    logger.info("Circuit analyzer initialized successfully. Application ready.")

logger.info("--"*60)
logger.info("--"*60)

# Create containers for results
if 'active_results' not in st.session_state:
    st.session_state.active_results = {
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': None,
        'uploaded_file_type': None,
        'uploaded_file_name': None,
        'annotated_image': None,
        'component_stats': None,
        'node_visualization': None,
        'node_mask': None,
        'enhanced_mask': None,
        'contour_image': None,
        'corners_image': None,
        'sam2_output': None,
        'valueless_netlist_text': None,
        'enum_img': None,
        'detailed_timings': {}
    }

if 'model_load_toast_shown' not in st.session_state:
    st.session_state.model_load_toast_shown = False

# Track previous upload to prevent unnecessary resets
if 'previous_upload_name' not in st.session_state:
    st.session_state.previous_upload_name = None

# Track analysis state
if 'analysis_in_progress' not in st.session_state:
    st.session_state.analysis_in_progress = False
if 'start_analysis_triggered' not in st.session_state:
    st.session_state.start_analysis_triggered = False

# Add new session state variable after the other session state declarations
if 'final_netlist_generated' not in st.session_state:
    st.session_state.final_netlist_generated = False
if 'editable_netlist_content' not in st.session_state:
    st.session_state.editable_netlist_content = None

# Main content  
st.image("static/images/sdp_banner.png", use_container_width=True)

# File upload section
file_upload_container = st.container()
with file_upload_container:
    uploaded_file = st.file_uploader(
        "Drag and drop your circuit diagram here",
        type=['png', 'jpg', 'jpeg'],
        help="For best results, use a clear image"
    )

# Show SAM2 status
if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2:
    if not st.session_state.get('model_load_toast_shown', False):
        st.toast("✅ Model loaded successfully")
        st.session_state.model_load_toast_shown = True
else:
    st.warning("⚠️ Model loading failed")

# Process uploaded file
if uploaded_file is not None:
    # Generate a consistent image path for the uploaded file
    disk_image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    
    # Only reset results if a new file is uploaded
    if st.session_state.get('previous_upload_name') != uploaded_file.name:
        st.session_state.previous_upload_name = uploaded_file.name
        
        # Log a clear separator for debugging
        logger.info("="*80)
        logger.info(f"NEW IMAGE UPLOADED: {uploaded_file.name}")
        logger.info("="*80)
        
        # Convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        
        # Clear results when new image is uploaded
        st.session_state.active_results = {
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
            'corners_image': None,
            'sam2_output': None,
            'valueless_netlist_text': None,
            'enum_img': None,
            'detailed_timings': {},
            'image_path': disk_image_path
        }
        
        # Reset the final netlist generation flag
        st.session_state.final_netlist_generated = False
        
        # Store original image
        st.session_state.active_results['original_image'] = image
        
        # Clear and save files
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save using PIL to preserve EXIF data
        pil_image = Image.open(uploaded_file)
            
        # Auto-rotate the image based on EXIF orientation tag
        logger.info(f"Checking EXIF orientation for {uploaded_file.name}")
        try:
            exif = pil_image._getexif()
            if exif and 0x0112 in exif:  # 0x0112 is the orientation tag ID
                orientation = exif[0x0112]
                logger.info(f"Found orientation tag: {orientation}")
                if orientation != 1:  # If orientation is not normal
                    logger.info(f"Auto-rotating image with orientation {orientation}")
                    pil_image = ImageOps.exif_transpose(pil_image)
                    # Update the numpy array image as well
                    image = np.array(pil_image)
                    st.session_state.active_results['original_image'] = image
        except Exception as e:
            logger.error(f"Error checking/rotating image based on EXIF: {str(e)}")
        
        # Save the processed image, preserving EXIF data
        exif_bytes = pil_image.info.get('exif')
        if exif_bytes:
            pil_image.save(disk_image_path, format=pil_image.format, exif=exif_bytes)
            logger.info(f"Saved image {disk_image_path} with EXIF data ({len(exif_bytes)} bytes).")
        else:
            pil_image.save(disk_image_path, format=pil_image.format)
            logger.info(f"Saved image {disk_image_path} without EXIF data (EXIF not found in PIL image after processing).")
        
        # Automatically trigger analysis
        st.session_state.start_analysis_triggered = True
        st.session_state.analysis_in_progress = True


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

# Analysis logic, executed if triggered
if st.session_state.get('start_analysis_triggered', False):
    st.session_state.start_analysis_triggered = False # Consume the trigger
    
    # Ensure an image is loaded and ready for analysis
    if st.session_state.active_results.get('original_image') is None or \
       st.session_state.active_results.get('image_path') is None:
        logger.warning("Analysis triggered, but required image data (original_image or image_path) is missing in active_results.")
        st.session_state.analysis_in_progress = False # Reset if it was set True by trigger
        # No st.rerun() here to avoid potential loops if the state isn't fixed by other flows.
        # The UI will update based on the rerun that triggered this or subsequent user actions.
    else:
        try:
            # analysis_in_progress is already True if set by the trigger.
            
            # Create a placeholder for the custom loader and text
            loader_placeholder = st.empty()

            with loader_placeholder.container():
                st.markdown("""
                    <div class="loader-popup-overlay">
                        <div class="loader-popup-content">
                            <div class="loader"></div>
                            <p style="text-align: center; margin-top: 15px; font-size: 1.1em; color: #333;">please wait ...</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Start timing the analysis
                overall_start_time = time.time()
                detailed_timings = {}
                logger.info("Starting complete circuit analysis...")
                
                # Step 1A: Initial Component Detection (YOLO on original image)
                step_start_time_yolo = time.time()
                raw_bboxes_orig_coords = []
                # Ensure original_image is available (already checked above, but good for clarity)
                if st.session_state.active_results['original_image'] is not None:
                    logger.debug("Step 1A: Detecting components with YOLO on original image...")
                    raw_bboxes_orig_coords = analyzer.bboxes(st.session_state.active_results['original_image'])
                    logger.debug(f"Detected {len(raw_bboxes_orig_coords)} raw bounding boxes on original image")
                    
                    # Apply Non-Maximum Suppression to bboxes in original coordinates
                    bboxes_orig_coords_nms = non_max_suppression_by_confidence(raw_bboxes_orig_coords, iou_threshold=0.6)
                    logger.debug(f"After NMS: {len(bboxes_orig_coords_nms)} bounding boxes on original image")
                    st.session_state.active_results['bboxes_orig_coords_nms'] = bboxes_orig_coords_nms
                else:
                    # This case should be prevented by the check at the start of this 'if start_analysis_triggered' block
                    logger.error("No original image found for YOLO detection during analysis.")
                    st.error("Error: Original image not available for analysis.")
                    st.session_state.analysis_in_progress = False
                    loader_placeholder.empty() # Clear loader
                    st.stop() # Stop further execution in this problematic state
                detailed_timings['YOLO Component Detection'] = time.time() - step_start_time_yolo

                # Step 1AA: Enrich BBoxes with Semantic Directions from LLaMA (Groq)
                # This step should happen after initial YOLO bbox detection and NMS,
                # using the bboxes in original image coordinates (bboxes_orig_coords_nms)
                # and the original image (st.session_state.active_results['original_image'], which should be RGB).
                can_enrich = (
                    st.session_state.active_results.get('bboxes_orig_coords_nms') and
                    st.session_state.active_results.get('original_image') is not None and
                    hasattr(analyzer, 'groq_client') and analyzer.groq_client # Check if attr exists and then its truthiness
                )
                if can_enrich:
                    logger.info("Attempting to enrich component bboxes with semantic directions from LLaMA...")
                    step_start_time_llama_enrich = time.time()
                    try:
                        # The _enrich_bboxes_with_directions method modifies bboxes_orig_coords_nms in-place.
                        # These bboxes (now potentially with 'semantic_direction') will then be used in cropping
                        # and the 'semantic_direction' attribute should be preserved by deepcopy in crop_image_and_adjust_bboxes.
                        analyzer._enrich_bboxes_with_directions(
                            st.session_state.active_results['original_image'],
                            st.session_state.active_results['bboxes_orig_coords_nms']
                        )
                        logger.info("Semantic direction enrichment step completed.")
                        detailed_timings['LLaMA Direction Enrichment'] = time.time() - step_start_time_llama_enrich
                    except Exception as e_enrich:
                        logger.error(f"Error during LLaMA semantic direction enrichment: {e_enrich}")
                        st.warning("Could not determine semantic directions for some components using LLaMA.")
                        detailed_timings['LLaMA Direction Enrichment'] = time.time() - step_start_time_llama_enrich # Log time even on failure
                # If enrichment couldn't happen, log a warning if the basic prerequisites were met but groq_client was the issue.
                elif (
                    st.session_state.active_results.get('bboxes_orig_coords_nms') and
                    st.session_state.active_results.get('original_image') is not None
                ):
                    # This implies can_enrich was false, and since bboxes & image are present, 
                    # it was likely due to groq_client issues (either not an attribute or evaluates to False).
                    logger.warning("Skipping LLaMA semantic direction enrichment: Groq client not available, analyzer instance might be outdated, or GROQ_API_KEY missing.")

                # Step 1B: SAM2 Segmentation (on original image) & Get SAM2 Extent
                step_start_time_sam = time.time()
                full_binary_sam_mask, sam2_colored_display_output, sam_extent_bbox = None, None, None
                if st.session_state.active_results['original_image'] is not None and analyzer.use_sam2:
                    logger.debug("Step 1B: Performing SAM2 segmentation on original image...")
                    full_binary_sam_mask, sam2_colored_display_output, sam_extent_bbox = analyzer.segment_with_sam2(
                        st.session_state.active_results['original_image']
                    )
                    st.session_state.active_results['sam2_output'] = sam2_colored_display_output # For display
                    if sam_extent_bbox:
                        logger.info(f"SAM2 extent bbox found: {sam_extent_bbox}")
                    else:
                        logger.warning("SAM2 did not return a valid extent bbox. Cropping will be skipped.")
                elif not analyzer.use_sam2:
                    logger.warning("SAM2 is disabled. Cropping based on SAM2 extent will be skipped.")
                detailed_timings['SAM2 Segmentation & Extent'] = time.time() - step_start_time_sam

                # Step 1C: Crop based on SAM2 Extent
                step_start_time_crop = time.time()
                image_for_analysis = st.session_state.active_results['original_image'] # Default
                bboxes_for_analysis = st.session_state.active_results.get('bboxes_orig_coords_nms', [])
                cropped_sam_mask_for_nodes = full_binary_sam_mask # Default to full if no crop
                crop_details_returned = None

                if sam_extent_bbox and full_binary_sam_mask is not None:
                    logger.debug("Attempting to crop image and SAM2 mask based on SAM2 extent...")
                    cropped_visual_image, adjusted_yolo_bboxes, crop_details_returned = analyzer.crop_image_and_adjust_bboxes(
                        st.session_state.active_results['original_image'],
                        st.session_state.active_results['bboxes_orig_coords_nms'],
                        sam_extent_bbox,
                        padding=80 # Added some padding
                    )
                    if crop_details_returned:
                        image_for_analysis = cropped_visual_image
                        bboxes_for_analysis = adjusted_yolo_bboxes
                        # Crop the full_binary_sam_mask using crop_details_returned
                        crop_x, crop_y, crop_w, crop_h = crop_details_returned
                        cropped_sam_mask_for_nodes = full_binary_sam_mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                        logger.info(f"Visual image and SAM2 mask cropped. Cropped SAM mask shape: {cropped_sam_mask_for_nodes.shape}")
                        st.session_state.active_results['crop_details'] = crop_details_returned
                    else:
                        logger.warning("Cropping based on SAM2 extent failed or was not possible. Using originals.")
                else:
                    logger.info("Skipping crop: SAM2 extent bbox or full mask not available.")
                
                st.session_state.active_results['image_for_analysis'] = image_for_analysis
                st.session_state.active_results['bboxes'] = bboxes_for_analysis # These are primary for node analysis & enum
                st.session_state.active_results['cropped_sam_mask_for_nodes'] = cropped_sam_mask_for_nodes
                detailed_timings['Image Cropping'] = time.time() - step_start_time_crop

                # Step 1D: Create Annotated image for display (on image_for_analysis)
                annotated_display_image = create_annotated_image(
                    image_for_analysis, 
                    bboxes_for_analysis 
                )
                st.session_state.active_results['annotated_image'] = annotated_display_image
                
                # Component Statistics from bboxes_for_analysis
                component_stats = calculate_component_stats(bboxes_for_analysis)
                st.session_state.active_results['component_stats'] = component_stats
                # Combined timing for what was previously just "Component Detection"
                # This now includes YOLO, SAM2-Extent, Cropping, and Annotation for display.
                # For a more direct comparison, sum them up or keep separate as done with detailed_timings.

                # Step 2: Node Analysis (using cropped SAM2 mask and adjusted bboxes)
                step_start_time_nodes = time.time()
                if bboxes_for_analysis is not None and cropped_sam_mask_for_nodes is not None and analyzer.use_sam2:
                    try:
                        logger.debug("Step 2: Analyzing node connections using cropped SAM mask and adjusted bboxes...")
                        nodes, emptied_mask, enhanced, contour_image, final_visualization = analyzer.get_node_connections(
                            image_for_analysis,             # Cropped visual image (for context)
                            cropped_sam_mask_for_nodes,   # Cropped SAM2 binary mask (for processing)
                            bboxes_for_analysis           # YOLO bboxes relative to cropped images
                        )
                        logger.debug(f"Node analysis completed: {len(nodes)} nodes identified")
                        st.session_state.active_results['nodes'] = nodes
                        st.session_state.active_results['node_visualization'] = final_visualization
                        # The following are debug/intermediate images from node analysis, based on its internal processing scale
                        st.session_state.active_results['node_mask'] = emptied_mask       # This is based on cropped_sam_mask_for_nodes after component removal
                        st.session_state.active_results['enhanced_mask'] = enhanced     # Based on resized emptied_mask
                        st.session_state.active_results['contour_image'] = contour_image  # Based on enhanced mask

                    except Exception as e:
                        error_msg = f"Error during node analysis: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
                        logger.warning("Continuing execution despite node analysis error")
                elif not analyzer.use_sam2:
                    logger.warning("Node analysis skipped: SAM2 is disabled.")
                    st.warning("Node analysis cannot be performed as SAM2 is disabled.")
                else:
                    logger.error("Node analysis skipped: Bounding boxes or cropped SAM2 mask not available.")
                    st.error("Components or circuit mask not ready for node analysis.")
                detailed_timings['Node Analysis'] = time.time() - step_start_time_nodes
                
                # Step 3: Generate Netlist
                step_start_time_netlist = time.time()
                if st.session_state.active_results.get('nodes') is not None:
                    try:
                        logger.debug("Step 3: Generating initial netlist...")
                        valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.active_results['nodes'])
                        valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                        logger.debug(f"Generated initial netlist with {len(valueless_netlist)} components")
                        
                        st.session_state.active_results['netlist'] = valueless_netlist
                        st.session_state.active_results['valueless_netlist_text'] = valueless_netlist_text
                        st.session_state.active_results['netlist_text'] = valueless_netlist_text
                        st.session_state.editable_netlist_content = valueless_netlist_text
                        
                        # --- BEGIN: Generate initial netlist WITHOUT LLaMA directions for comparison ---
                        try:
                            logger.debug("Generating initial netlist WITHOUT LLaMA directions for comparison...")
                            nodes_copy_for_no_llama = deepcopy(st.session_state.active_results['nodes'])
                            for node_data in nodes_copy_for_no_llama:
                                for component_in_node in node_data.get('components', []):
                                    # Neutralize semantic_direction to force default ordering
                                    component_in_node['semantic_direction'] = "UNKNOWN" # or None
                            
                            valueless_netlist_no_llama_dir = analyzer.generate_netlist_from_nodes(nodes_copy_for_no_llama)
                            valueless_netlist_text_no_llama_dir = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist_no_llama_dir])
                            st.session_state.active_results['valueless_netlist_text_no_llama_dir'] = valueless_netlist_text_no_llama_dir
                            logger.debug(f"Generated initial netlist WITHOUT LLaMA directions: {len(valueless_netlist_no_llama_dir)} components")
                        except Exception as e_no_llama:
                            logger.error(f"Error generating netlist without LLaMA directions: {e_no_llama}")
                            st.session_state.active_results['valueless_netlist_text_no_llama_dir'] = "Error generating this version."
                        # --- END: Generate initial netlist WITHOUT LLaMA directions ---

                        logger.debug("Enumerating components for later Gemini labeling on (potentially cropped) image_for_analysis...")
                        enum_img, bbox_ids_with_visual_enum = analyzer.enumerate_components(
                            image_for_analysis, # This is the potentially SAM2-extent cropped visual image
                            deepcopy(bboxes_for_analysis) # These bboxes are relative to image_for_analysis and have persistent_uid
                        )
                        st.session_state.active_results['enum_img'] = enum_img # Potentially cropped and enumerated image for Gemini
                        st.session_state.active_results['bbox_ids'] = bbox_ids_with_visual_enum
                        
                    except Exception as netlist_error:
                        error_msg = f"Error generating initial netlist: {str(netlist_error)}"
                        logger.error(error_msg)
                        st.error(error_msg)
                else:
                    logger.warning("No nodes found for netlist generation")
                    st.warning("Could not identify connection nodes in the circuit. The netlist may be incomplete.")
                    
                    # Still try to generate a basic netlist from bboxes if available
                    if st.session_state.active_results['bboxes'] is not None:
                        try:
                            logger.debug("Attempting to generate basic netlist from components only...")
                            # Create empty nodes if needed
                            empty_nodes = []
                            valueless_netlist = analyzer.generate_netlist_from_nodes(empty_nodes)
                            st.session_state.active_results['netlist'] = valueless_netlist
                            st.session_state.active_results['netlist_text'] = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                            st.session_state.active_results['valueless_netlist_text'] = st.session_state.active_results['netlist_text']
                        except Exception as fallback_error:
                            logger.error(f"Error generating fallback netlist: {str(fallback_error)}")
                
                detailed_timings['Netlist Generation'] = time.time() - step_start_time_netlist
                
                # Log analysis summary
                if st.session_state.active_results.get('netlist') and log_level in ['DEBUG', 'INFO']:
                    try:
                        component_counts = {}
                        for line in st.session_state.active_results['netlist']:
                            comp_type = line['class']
                            if comp_type not in component_counts:
                                component_counts[comp_type] = 0
                            component_counts[comp_type] += 1
                        
                        logger.info("Analysis results summary:")
                        logger.info(f"- Image: {st.session_state.active_results.get('uploaded_file_name', 'Unknown')}")
                        logger.info(f"- Total components detected: {len(st.session_state.active_results['netlist'])}")
                        for comp_type, count in component_counts.items():
                            logger.info(f"  - {comp_type}: {count}")
                        if st.session_state.active_results.get('nodes'):
                            logger.info(f"- Total nodes: {len(st.session_state.active_results['nodes'])}")
                    except Exception as summary_error:
                        logger.error(f"Error generating analysis summary: {str(summary_error)}")
                
                # End timing and log the duration
                overall_end_time = time.time()
                elapsed_time = overall_end_time - overall_start_time
                logger.info(f"Circuit analysis completed in {elapsed_time:.2f} seconds")
                
                # Store elapsed time and detailed timings in active results
                st.session_state.active_results['elapsed_time'] = elapsed_time
                st.session_state.active_results['detailed_timings'] = detailed_timings

                # Clear the custom loader and text
                loader_placeholder.empty()
                
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            st.error(error_msg)
            
            # Clear the loader if it exists
            try:
                loader_placeholder.empty()
            except:
                pass
        finally:
            # Always reset analysis state when done, even if there was an error
            st.session_state.analysis_in_progress = False
            st.rerun() # Ensures UI update after analysis completion or error
    
# Display results section after analysis
if st.session_state.active_results['original_image'] is not None:
    # Check if the current analysis is based on a carousel image or an uploaded file
    # to correctly display image details (EXIF might not apply to all carousel examples if they are processed)
    current_image_path = st.session_state.active_results.get('image_path')
    is_uploaded_file = current_image_path and UPLOAD_DIR in current_image_path

    # Step 1: Image and Component Detection side by side
    st.markdown("## 📊 Analysis Results")
    
    # Display analysis time if available
    if 'elapsed_time' in st.session_state.active_results:
        if 'detailed_timings' in st.session_state.active_results and st.session_state.active_results['detailed_timings']:
            expander_label = f"✅ Analysis completed in {st.session_state.active_results['elapsed_time']:.2f} seconds - Click to see details"
            st.markdown("<div class='detailed-timings-expander'>", unsafe_allow_html=True)
            with st.expander(expander_label):
                for step, duration in st.session_state.active_results['detailed_timings'].items():
                    st.markdown(f"- **{step}**: {duration:.2f} seconds")
            st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Image")
        st.image(st.session_state.active_results['original_image'], use_container_width=True)
        
        # Show image details as a dropdown
        with st.expander("Image Details"):
            # Basic image properties that are always available
            img = Image.open(current_image_path) # Use current_image_path
            h, w = st.session_state.active_results['original_image'].shape[:2]
            st.markdown("### Basic Properties")
            st.markdown(f"- **Size**: {w}x{h} pixels")
            st.markdown(f"- **Format**: {img.format}")
            st.markdown(f"- **Mode**: {img.mode}")
            if 'uploaded_file_name' in st.session_state.active_results:
                st.markdown(f"- **Name**: {st.session_state.active_results['uploaded_file_name']}")
            
            # EXIF data section
            st.markdown("### EXIF Data")
            try:
                exif = format_exif_data(current_image_path) # Use current_image_path
                if exif:
                    # Show important EXIF data in a clean table format
                    table_rows = ["| Tag | Value |", "| --- | --- |"]
                    
                    # Add all important tags to the table
                    for tag, value in sorted(exif.items()):
                        # Escape pipe characters for markdown tables
                        if isinstance(value, str):
                            value = value.replace('|', '\\|')
                        if isinstance(tag, str):
                            tag = tag.replace('|', '\\|')
                            
                        table_rows.append(f"| {tag} | {value} |")
                        
                    if len(table_rows) > 2:
                        st.markdown("\n".join(table_rows))
                    else:
                        st.info("No important EXIF tags found in this image.")
                else:
                    st.info("ℹ️ This image does not contain any EXIF metadata. This is normal for images that have been processed, screenshots, or images saved without preserving metadata.")
            except Exception as e:
                st.warning(f"Could not read EXIF data: {str(e)}")
                logger.error(f"Error displaying EXIF data: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    with col2:
        st.markdown("### Component Detection")
        # Display the annotated image (which is based on image_for_analysis - potentially cropped)
        if st.session_state.active_results['annotated_image'] is not None:
            st.image(st.session_state.active_results['annotated_image'], use_container_width=True)
            
            # Show component statistics in a dropdown
            with st.expander("Component Statistics"):
                if st.session_state.active_results['component_stats'] is not None:
                    # Create a dataframe for better organization
                    stats_data = []
                    for component, stats in st.session_state.active_results['component_stats'].items():
                        avg_conf = stats['total_conf'] / stats['count']
                        stats_data.append({
                            "Component": component,
                            "N": stats['count'],
                            "Avg Conf": f"{avg_conf:.2f}"
                        })
                    
                    if stats_data:
                        st.table(stats_data)
            
            # Expander for LLaMA Stage 1 (Direction) Debug Output
            with st.expander("Debug: LLaMA Directions"):
                if 'bboxes' in st.session_state.active_results and st.session_state.active_results['bboxes']:
                    directions_data = []
                    for comp_bbox in st.session_state.active_results['bboxes']:
                        if 'semantic_direction' in comp_bbox and comp_bbox['semantic_direction'] is not None:
                            directions_data.append({
                                "Persistent UID": comp_bbox.get('persistent_uid', 'N/A'),
                                "YOLO Class": comp_bbox.get('class', 'N/A'),
                                "Semantic Direction": comp_bbox['semantic_direction']
                            })
                    if directions_data:
                        st.table(directions_data)
                    else:
                        st.info("No semantic directions were determined or available in the processed bboxes.")
                else:
                    st.info("Processed component bounding boxes ('bboxes') not available in session state.")
        
        else:
            st.info("Run analysis to see component detection")
    
    # Step 2: Node Analysis
    if st.session_state.active_results['contour_image'] is not None or st.session_state.active_results['sam2_output'] is not None:
        st.markdown("## 🔗 Node Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Final Node Connections")
            if st.session_state.active_results['node_visualization'] is not None:
                st.image(st.session_state.active_results['node_visualization'], use_container_width=True)
        
        with col2:
            st.markdown("### SAM2 Segmentation")
            if st.session_state.active_results['sam2_output'] is not None:
                st.image(st.session_state.active_results['sam2_output'], use_container_width=True)
        
        # Show additional debug images in a dropdown
        with st.expander("Additional Debug Images"):
            debug_col1, debug_col2 = st.columns(2)
            
            with debug_col1:
                if st.session_state.active_results['node_mask'] is not None:
                    st.image(st.session_state.active_results['node_mask'], caption="Emptied Mask", use_container_width=True)
            
            with debug_col2:
                if st.session_state.active_results['enhanced_mask'] is not None:
                    st.image(st.session_state.active_results['enhanced_mask'], caption="Enhanced Mask", use_container_width=True)
            
            if st.session_state.active_results['contour_image'] is not None:
                st.image(st.session_state.active_results['contour_image'], caption="Contours", use_container_width=True)
    
    # Step 3: Netlist
    if st.session_state.active_results.get('netlist_text') is not None:
        # Modified layout to align column sizes better
        st.markdown("## 📝 Circuit Netlist")
        
        # Create columns for button and loader with a better ratio
        btn_col, loader_col = st.columns([4, 1])
        
        # Move button to the first column
        with btn_col:
            netlist_btn = st.button("Get Final Netlist", use_container_width=True, type="primary", disabled=st.session_state.final_netlist_generated)
        
        # Add the loader in the second column
        with loader_col:
            # Create a container for the loader with better vertical alignment
            if netlist_btn:
                st.markdown('<div style="display:flex; align-items:center; justify-content:center; height:100%; min-height:38px;"><div class="gemini-loader"></div></div>', unsafe_allow_html=True)
            else:
                # Empty container to maintain layout when loader is not shown
                st.markdown('<div style="height:38px;"></div>', unsafe_allow_html=True)
        
        # Process the button click
        if netlist_btn and not st.session_state.final_netlist_generated:
            try:
                final_start_time = time.time()
                
                # Get the stored data needed for final netlist generation
                # valueless_netlist has component entries which include bboxes with persistent_uid
                valueless_netlist = st.session_state.active_results['netlist']
                enum_img = st.session_state.active_results['enum_img'] # This is the (potentially cropped) image with numbers
                # bbox_ids are the bboxes from (potentially cropped) image, now with visual enumeration 'id' and 'persistent_uid'
                bbox_ids_for_fix = st.session_state.active_results['bbox_ids'] 
                
                # Create deep copy for the final netlist
                netlist = deepcopy(valueless_netlist)
                
                # Call Gemini for component labeling
                try:
                    logger.debug("Calling Gemini for component labeling using (potentially cropped) enumerated image...")
                    # gemini_labels_openrouter receives the (potentially cropped) enum_img
                    gemini_info = gemini_labels_openrouter(enum_img) 
                    logger.debug(f"Received information for {len(gemini_info) if gemini_info else 0} components from Gemini")
                    logger.info(f"GEMINI BARE OUTPUT (gemini_info): {gemini_info}")
                    st.session_state.active_results['vlm_stage2_output'] = gemini_info # Store for debugging
                    
                    # fix_netlist will correlate gemini_info (using visual enum 'id') 
                    # with netlist lines (via persistent_uid found in bbox_ids_for_fix)
                    analyzer.fix_netlist(netlist, gemini_info, bbox_ids_for_fix)
                except Exception as gemini_error:
                    logger.error(f"Error calling Gemini API: {str(gemini_error)}")
                    st.warning(f"Could not get component values from AI: {str(gemini_error)}. Using basic netlist.")
                    # Still use the valueless netlist
                    netlist = valueless_netlist
                
                # Debug: Log the state of netlist lines before stringifying
                if logger.isEnabledFor(logging.DEBUG):
                    for i, ln_debug in enumerate(netlist):
                        logger.debug(f"App.py netlist line {i} before stringify: {ln_debug}")
                
                # Filter out lines with None values before generating the final string
                netlist = [line for line in netlist if line.get('value') is not None and str(line.get('value')).strip().lower() != 'none']

                # Generate the final netlist text
                netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])
                
                # Store the results
                st.session_state.active_results['netlist'] = netlist
                st.session_state.active_results['netlist_text'] = netlist_text
                st.session_state.editable_netlist_content = netlist_text
                
                # Calculate and store the final netlist generation time
                final_elapsed_time = time.time() - final_start_time
                if 'detailed_timings' in st.session_state.active_results:
                    st.session_state.active_results['detailed_timings']['Final Netlist Generation'] = final_elapsed_time
                
                # Mark as completed
                st.session_state.final_netlist_generated = True
                
                # Show success message
                st.success("Final netlist generated successfully!")
                
                # Force a rerun to update the UI
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating final netlist: {str(e)}")
                logger.error(f"Error generating final netlist: {str(e)}")
        
        # Main netlist display: Initial (LLaMA) and Final side-by-side
        col1_main_netlist, col2_main_netlist = st.columns(2)
        with col1_main_netlist:
            st.markdown("### Initial Netlist") # (This is with LLaMA directions)
            if 'valueless_netlist_text' in st.session_state.active_results:
                st.code(st.session_state.active_results['valueless_netlist_text'], language="rust")
            else:
                st.info("Initial netlist not available.")

        with col2_main_netlist:
            st.markdown("### Final Netlist") # (After VLM Stage 2)
            if st.session_state.final_netlist_generated and 'netlist_text' in st.session_state.active_results:
                # MODIFIED: Use st.code for displaying the final netlist
                st.code(st.session_state.active_results['netlist_text'], language="rust")
            elif not st.session_state.final_netlist_generated:
                st.info("Click 'Get Final Netlist' button above to generate.")
            else:
                st.info("Final netlist not available.")
        
        # Dropdown 1: Image Sent to Gemini/VLM
        with st.expander("🔍 Debug: VLM"):
            st.markdown("### Image Sent to VLM")
            if 'enum_img' in st.session_state.active_results and st.session_state.active_results['enum_img'] is not None:
                st.image(st.session_state.active_results['enum_img'], caption="Image sent for VLM Stage 2 analysis")
            else:
                st.info("Enumerated image for VLM not available.")
                
            st.markdown("### VLM Analysis Output")
            if 'vlm_stage2_output' in st.session_state.active_results and st.session_state.active_results['vlm_stage2_output']:
                gemini_info = st.session_state.active_results['vlm_stage2_output']
                # Format the raw output like a netlist
                formatted_output = "[\n"
                for comp in gemini_info:
                    formatted_output += "    {\n"
                    for key, value in comp.items():
                        formatted_output += f"        '{key}': '{value}',\n"
                    formatted_output = formatted_output.rstrip(',\n') + "\n    },\n"
                formatted_output = formatted_output.rstrip(',\n') + "\n]"
                
                st.code(formatted_output, language="python")
            else:
                st.info("No VLM analysis output available")

        # Dropdown 2: Differences in Initial Netlists (if any)
        netlist_llama_text = st.session_state.active_results.get('valueless_netlist_text')
        netlist_no_llama_text = st.session_state.active_results.get('valueless_netlist_text_no_llama_dir')
        has_llama = netlist_llama_text is not None
        has_no_llama = netlist_no_llama_text is not None
        are_different_initials = False
        if has_llama and has_no_llama and netlist_llama_text != netlist_no_llama_text:
            are_different_initials = True
        
        if are_different_initials:
            with st.expander("🔍 Differences in Initial Netlists (LLaMA vs. No LLaMA)"):
                st.markdown("Details of Differences:")
                lines_llama = netlist_llama_text.splitlines()
                lines_no_llama = netlist_no_llama_text.splitlines()
                num_lines_llama = len(lines_llama)
                num_lines_no_llama = len(lines_no_llama)
                max_l = max(num_lines_llama, num_lines_no_llama)
                
                any_line_diff_shown = False
                for i in range(max_l):
                    line_l_content = lines_llama[i] if i < num_lines_llama else None
                    line_nl_content = lines_no_llama[i] if i < num_lines_no_llama else None

                    if line_l_content != line_nl_content:
                        any_line_diff_shown = True
                        st.markdown(f"**Line {i+1}:**")
                        if line_l_content is not None:
                            st.markdown(f"  - With LLaMA:    `{line_l_content}`")
                        else:
                            st.markdown(f"  - With LLaMA:    *Line not present*")
                        
                        if line_nl_content is not None:
                            st.markdown(f"  - Without LLaMA: `{line_nl_content}`")
                        else:
                            st.markdown(f"  - Without LLaMA: *Line not present*")
                        st.markdown("") # Add a little space

                if not any_line_diff_shown:
                    st.markdown("The netlists are marked as different, but no specific line-by-line variances were rendered. This could be due to subtle differences like trailing whitespace. The initial netlist (with LLaMA) is shown above.")
        
    
    # Step 4: SPICE Analysis - keep as is
    # Only show SPICE analysis options AFTER the final netlist is generated
    if st.session_state.get('final_netlist_generated', False):
        st.markdown("## ⚡ SPICE Analysis")

        # Initialize session state for analysis_mode and ac_frequency if not present
        # analysis_mode will be overwritten by auto-detection.
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = "DC (.op)"
        if 'ac_frequency' not in st.session_state:
            st.session_state.ac_frequency = 60.0  # Default to 60Hz

        # Auto-determine analysis mode from the editable netlist content
        # This editable_netlist_content should be the one from the final VLM-processed netlist
        current_netlist_text_for_spice = st.session_state.get('editable_netlist_content', "")
        determined_analysis_mode = "DC (.op)" # Default to DC

        if current_netlist_text_for_spice:
            netlist_lines = current_netlist_text_for_spice.split('\n')
            # Pattern for "value:phase", e.g., "4:-45". Allows for optional whitespace.
            mag_phase_pattern = re.compile(r"^[+-]?\d*\.?\d+\s*:\s*[+-]?\d*\.?\d+$")

            for line in netlist_lines:
                line_strip = line.strip()
                if not line_strip or not line_strip[0].isalpha():
                    continue

                line_upper = line_strip.upper() # For checking component type and " AC "
                original_line_parts = line_strip.split()
                
                component_type = line_upper[0]
                is_ac_source_for_this_line = False
                
                if component_type == 'V' or component_type == 'I':
                    # 1. Check for " AC " keyword (common SPICE syntax for AC analysis part of source)
                    if " AC " in line_upper: # e.g., "V1 1 0 DC 5 AC 1 0" or "V1 1 0 AC 1 0"
                        is_ac_source_for_this_line = True
                    else:
                        # 2. Check for "magnitude:phase" pattern in the arguments
                        #    e.g., Vname n+ n- value. The value itself is "mag:phase".
                        #    V3 3 0 4:-45 -> original_line_parts = ['V3', '3', '0', '4:-45']
                        if len(original_line_parts) >= 4: # Minimum: Name N+ N- Value
                            # Check argument parts starting from where values are expected
                            for part_idx in range(3, len(original_line_parts)):
                                if mag_phase_pattern.fullmatch(original_line_parts[part_idx].strip()):
                                    is_ac_source_for_this_line = True
                                    break # Found mag:phase in this line's parts, break inner loop (parts)
                    
                    if is_ac_source_for_this_line:
                        determined_analysis_mode = "AC (.ac)"
                        break # Found an AC source line, break outer (netlist_lines) loop
        
        st.session_state.analysis_mode = determined_analysis_mode
        
        st.markdown(f"**Auto-detected Analysis Type:** `{st.session_state.analysis_mode}`")

        ac_frequency_hz = None
        if st.session_state.analysis_mode == "AC (.ac)":
            ac_frequency_hz = st.number_input(
                "Frequency for AC Analysis (Hz):",
                min_value=0.01, # Min value slightly above zero
                value=st.session_state.ac_frequency,
                step=100.0,
                format="%.2f",
                key="ac_freq_input"
            )
            st.session_state.ac_frequency = ac_frequency_hz


        if st.button("Run SPICE Analysis"):
            try: # General try for the button action
                if st.session_state.analysis_mode == "DC (.op)":
                    try:
                        if st.session_state.editable_netlist_content:
                            logger.debug("Running DC SPICE analysis with netlist:")
                            logger.debug(st.session_state.editable_netlist_content)
                            
                            net_text_dc = (
                                '.title detected_circuit_dc\n'
                                + st.session_state.editable_netlist_content
                                + '\n.end\n'
                            )
                            
                            # Show complete DC SPICE netlist
                            with st.expander("🔍 Debug: Complete SPICE Netlist (DC)"):
                                st.code(net_text_dc, language="python")
                            
                            logger.debug("Full DC SPICE netlist with control commands:")
                            logger.debug(net_text_dc)
                            
                            parser = SpiceParser(source=net_text_dc)
                            bootstrap_circuit = parser.build_circuit()
                            
                            logger.debug("Circuit elements (DC):")
                            for element in bootstrap_circuit.elements:
                                logger.debug(f"  {element}")
                            
                            simulator = bootstrap_circuit.simulator(
                                temperature=27, nominal_temperature=27, gmin=1e-12,
                                abstol=1e-12, reltol=1e-6, chgtol=1e-14,
                                trtol=7, itl1=100, itl2=50, itl4=10
                            )
                            
                            logger.debug("Running operating point analysis (DC)...")
                            analysis = simulator.operating_point()
                            
                            logger.debug("Raw DC analysis results:")
                            logger.debug("Nodes:")
                            for node, value in analysis.nodes.items():
                                logger.debug(f"  {node}: {value} (type: {type(value)})")
                            logger.debug("Branches:")
                            for branch, value in analysis.branches.items():
                                logger.debug(f"  {branch}: {value} (type: {type(value)})")
                            
                            node_voltages_dc = {}
                            for node, value in analysis.nodes.items():
                                try:
                                    voltage = float(value._value if hasattr(value, '_value_') else value.get_value() if hasattr(value, 'get_value') else value)
                                    node_voltages_dc[str(node)] = f"{voltage:.3f}V"
                                except Exception as e_val:
                                    logger.error(f"Error converting DC node voltage {node}: {str(e_val)}")
                                    node_voltages_dc[str(node)] = "Error"
                            
                            branch_currents_dc = {}
                            for branch, value in analysis.branches.items():
                                try:
                                    current = float(value._value if hasattr(value, '_value_') else value.get_value() if hasattr(value, 'get_value') else value)
                                    branch_currents_dc[str(branch)] = f"{current*1000:.3f}mA"
                                except Exception as e_val:
                                    logger.error(f"Error converting DC branch current {branch}: {str(e_val)}")
                                    branch_currents_dc[str(branch)] = "Error"
                            
                            col1_dc, col2_dc = st.columns(2)
                            with col1_dc:
                                st.markdown("### Node Voltages (DC)")
                                st.json(node_voltages_dc)
                            with col2_dc:
                                st.markdown("### Branch Currents (DC)")
                                st.json(branch_currents_dc)
                        else:
                            st.error("Netlist is empty. Please generate or edit the netlist before running DC SPICE analysis.")
                    
                    except Exception as e_dc:
                        error_msg_dc = f"❌ DC SPICE Analysis Error: {str(e_dc)}"
                        logger.error(error_msg_dc, exc_info=True)
                        st.error(error_msg_dc)

                elif st.session_state.analysis_mode == "AC (.ac)":
                    try:
                        if st.session_state.active_results.get('netlist'):
                            sim_netlist_data = deepcopy(st.session_state.active_results['netlist'])
                            spice_body_lines = []
                            
                            for line_dict in sim_netlist_data:
                                if line_dict.get('class') == 'gnd': # Skip gnd for explicit lines
                                    continue

                                original_value = str(line_dict.get('value', ''))
                                component_type_prefix = line_dict.get('component_type', '')

                                # Modified condition: Attempt AC parsing if it's a V or I source,
                                # as the VLM might provide AC-formatted values for components
                                # not strictly classified as 'voltage.ac' or 'current.ac' by YOLO.
                                if component_type_prefix in ['V', 'I']:
                                    parsed_params = _parse_vlm_ac_string(original_value)
                                    if parsed_params:
                                        # Revert to including explicit DC offset, as ngspice seems to prefer it.
                                        line_dict['value'] = f"{parsed_params['dc_offset']} AC {parsed_params['mag']} {parsed_params['phase']}"
                                        logger.debug(f"Processed AC source {line_dict.get('component_type','')}{line_dict.get('component_num','')}: original='{original_value}', spice_val='{line_dict['value']}'")
                                    else:
                                        # If parsing fails for a V or I source, and it wasn't already in 'AC ...' format,
                                        # it might be a DC value or an unparseable AC value. Log a warning but use original value
                                        # if it doesn't look like a malformed AC attempt (e.g. just numbers for DC).
                                        # If it explicitly started with "AC" or contained ":", but failed parsing, then warn and default.
                                        if original_value.lower().strip().startswith('ac') or ':' in original_value:
                                            default_ac_val = "0 AC 1 0" # Default: 0 DC offset, 1V/A mag, 0 phase
                                            st.warning(f"Could not parse AC parameters for {component_type_prefix}{line_dict.get('component_num','')}: '{original_value}'. Using default: {default_ac_val}")
                                            logger.warning(f"Could not parse AC parameters for {component_type_prefix}{line_dict.get('component_num','')}: '{original_value}'. Using default: {default_ac_val}")
                                            line_dict['value'] = default_ac_val
                                        # else: it's likely a DC value for a V/I source, leave it as is.
                                
                                # Use analyzer.stringify_line to build the SPICE line from the (potentially modified) line_dict
                                spice_line = analyzer.stringify_line(line_dict)
                                if spice_line: # stringify_line returns "" for gnd, which we already skipped
                                    spice_body_lines.append(spice_line)

                            netlist_content_for_pyspice = "\n".join(spice_body_lines)
                            
                            if not netlist_content_for_pyspice.strip():
                                 st.error("Netlist for AC analysis is effectively empty after processing. Cannot simulate.")
                            else:
                                net_text_ac = (
                                    '.title detected_circuit_ac\n'
                                    + netlist_content_for_pyspice
                                    + f'\n* Equivalent SPICE command being executed:\n* .ac lin 1 {ac_frequency_hz} {ac_frequency_hz}\n'
                                    + '\n.end\n'
                                )
                                
                                # Show complete AC SPICE netlist
                                with st.expander("🔍 Debug: Complete SPICE Netlist (AC)"):
                                    st.code(net_text_ac, language="python")
                                
                                logger.debug("Running AC SPICE analysis with netlist:")
                                logger.debug(net_text_ac)
                                
                                parser_ac = SpiceParser(source=net_text_ac)
                                circuit_ac = parser_ac.build_circuit()

                                logger.debug("Circuit elements (AC):")
                                for element in circuit_ac.elements:
                                    logger.debug(f"  {element}")

                                simulator_ac = circuit_ac.simulator(
                                    temperature=27, nominal_temperature=27, gmin=1e-12,
                                    abstol=1e-12, reltol=1e-6, chgtol=1e-14,
                                    trtol=7, itl1=100, itl2=50, itl4=10
                                )
                                
                                sim_freq_hz = st.session_state.ac_frequency
                                logger.debug(f"Running AC analysis at {sim_freq_hz} Hz...")
                                
                                # PySpice expects frequency with units for .ac()
                                # Changed to pass raw float in Hz as per PySpice documentation for .ac() method parameters
                                analysis_ac = simulator_ac.ac(
                                    variation='lin',
                                    start_frequency=u_Hz(sim_freq_hz),
                                    stop_frequency=u_Hz(sim_freq_hz),
                                    number_of_points=1
                                )
                                
                                logger.debug("Raw AC analysis results:")
                                logger.debug("Nodes:")
                                for node, value_waveform in analysis_ac.nodes.items():
                                    # For single point AC, waveform has 1 complex value
                                    complex_val = value_waveform[0] if len(value_waveform) > 0 else complex(0,0)
                                    logger.debug(f"  {node}: {complex_val} (type: {type(complex_val)})")
                                logger.debug("Branches:")
                                for branch, value_waveform in analysis_ac.branches.items():
                                    complex_val = value_waveform[0] if len(value_waveform) > 0 else complex(0,0)
                                    logger.debug(f"  {branch}: {complex_val} (type: {type(complex_val)})")

                                node_voltages_ac_display = {}
                                for node_name_ac, val_waveform_ac in analysis_ac.nodes.items():
                                    if len(val_waveform_ac) > 0:
                                        # Get the value and convert to complex safely
                                        raw_value = val_waveform_ac[0]
                                        print(f"DEBUG: Node {node_name_ac} raw value: {raw_value}, type: {type(raw_value)}")
                                        
                                        complex_voltage = safe_to_complex(raw_value)
                                        print(f"DEBUG: Converted to complex: {complex_voltage}")
                                        
                                        # Calculate magnitude and phase
                                        mag = np.abs(complex_voltage)
                                        phase = np.angle(complex_voltage, deg=True)
                                        node_voltages_ac_display[str(node_name_ac)] = f"{mag:.3f} ∠ {phase:.2f}° V"
                                    else:
                                        node_voltages_ac_display[str(node_name_ac)] = "Error (no data)"
                                
                                branch_currents_ac_display = {}
                                for branch_name, val_waveform in analysis_ac.branches.items():
                                    if len(val_waveform) > 0:
                                        # Get the value and convert to complex safely
                                        raw_value = val_waveform[0]
                                        print(f"DEBUG: Branch {branch_name} raw value: {raw_value}, type: {type(raw_value)}")
                                        
                                        complex_current = safe_to_complex(raw_value)
                                        print(f"DEBUG: Converted to complex: {complex_current}")
                                        
                                        # Calculate magnitude and phase
                                        mag = np.abs(complex_current)
                                        phase = np.angle(complex_current, deg=True)
                                        branch_currents_ac_display[str(branch_name)] = f"{mag:.3f} ∠ {phase:.2f}° A"
                                    else:
                                        branch_currents_ac_display[str(branch_name)] = "Error (no data)"


                                col1_ac, col2_ac = st.columns(2)
                                with col1_ac:
                                    st.markdown("### Node Voltages (AC)")
                                    st.json(node_voltages_ac_display)
                                with col2_ac:
                                    st.markdown("### Branch Currents (AC)")
                                    st.json(branch_currents_ac_display)

                                # After the AC analysis results display, add plots
                                st.markdown("### AC Analysis Plots")
                                
                                # Create tabs for different plot types
                                plot_tabs = st.tabs(["Phasor Diagram"])
                                
                                with plot_tabs[0]:
                                    # Phasor diagram for voltages and currents
                                    try:
                                        # Create figure with two subplots side by side
                                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})
                                        
                                        # Plot voltage phasors
                                        max_v_mag = 0
                                        for node, val_waveform in analysis_ac.nodes.items():
                                            if len(val_waveform) > 0 and str(node) != '0':  # Skip ground node
                                                complex_val = safe_to_complex(val_waveform[0])
                                                mag = np.abs(complex_val)
                                                angle = np.angle(complex_val)
                                                max_v_mag = max(max_v_mag, mag)
                                                # Plot phasor
                                                ax1.plot([0, angle], [0, mag], label=f'V({node})', 
                                                        marker='o', linewidth=2)
                                        
                                        # Adjust voltage plot
                                        ax1.set_title('Voltage Phasors')
                                        ax1.set_rmax(max_v_mag * 1.2)  # Add 20% margin
                                        ax1.grid(True)
                                        ax1.legend()
                                        
                                        # Plot current phasors
                                        max_i_mag = 0
                                        for branch, val_waveform in analysis_ac.branches.items():
                                            if len(val_waveform) > 0:
                                                complex_val = safe_to_complex(val_waveform[0])
                                                mag = np.abs(complex_val)
                                                angle = np.angle(complex_val)
                                                max_i_mag = max(max_i_mag, mag)
                                                # Plot phasor
                                                ax2.plot([0, angle], [0, mag], label=str(branch), 
                                                        marker='o', linewidth=2)
                                        
                                        # Adjust current plot
                                        ax2.set_title('Current Phasors')
                                        ax2.set_rmax(max_i_mag * 1.2)  # Add 20% margin
                                        ax2.grid(True)
                                        ax2.legend()
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close()
                                        
                                    except Exception as e_plot:
                                        st.error(f"Error generating phasor plots: {str(e_plot)}")

                    except Exception as e_ac:
                        error_msg_ac = f"❌ AC SPICE Analysis Error: {str(e_ac)}"
                        logger.error(error_msg_ac, exc_info=True)
                        st.error(error_msg_ac)
                        st.info("💡 Tip: Check AC source parameters, frequency, and circuit connectivity.")
            
            # This except block is for the general try block associated with the st.button action
            except Exception as e: 
                error_msg = f"❌ SPICE Analysis Main Error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                st.info("💡 Tip: An unexpected error occurred during SPICE analysis setup.")
    # If final_netlist_generated is False, but an initial netlist (valueless) exists
    elif st.session_state.active_results.get('netlist_text') is not None:
        st.markdown("## ⚡ SPICE Analysis")
        st.info("Please click 'Get Final Netlist' above to enable SPICE analysis and component value detection.")