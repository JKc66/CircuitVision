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
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

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
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        
        # If there's no EXIF data or no orientation tag, return the original image
        if not exif or 0x0112 not in exif:  # 0x0112 is the orientation tag ID
            return image
        
        orientation = exif[0x0112]
        logger.info(f"Auto-rotating image with orientation tag: {orientation}")
        
        # Apply rotation based on orientation
        rotated_image = ImageOps.exif_transpose(image)
        return rotated_image
    except Exception as e:
        logger.error(f"Error auto-rotating image: {str(e)}")
        return Image.open(image_path)  # Return original on error

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
    try:
        img = Image.open(image_path)
        
        # Define important tags we want to show
        important_tags = {
            'Software',
            'Orientation'
        }
        
        # Get EXIF data
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

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"YOLO model path: {YOLO_MODEL_PATH}")
logger.info(f"SAM2 config path: {SAM2_CONFIG_PATH}")
logger.info(f"SAM2 base checkpoint path: {SAM2_BASE_CHECKPOINT_PATH}")
logger.info(f"SAM2 finetuned checkpoint path: {SAM2_FINETUNED_CHECKPOINT_PATH}")


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

# Create containers for results
if 'active_results' not in st.session_state:
    st.session_state.active_results = {
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': None,
        'uploaded_file_type': None, # Added to store file type
        'uploaded_file_name': None, # Added to store file name
        'annotated_image': None,
        'component_stats': None,
        'node_visualization': None,
        'node_mask': None,
        'enhanced_mask': None,
        'contour_image': None,
        'corners_image': None,
        'sam2_output': None,
        'valueless_netlist_text': None, # Added for consistency
        'enum_img': None, # Added for consistency
        'detailed_timings': {} # Added for detailed timings
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

# Add new session state variable after the other session state declarations (around line 236)
if 'final_netlist_generated' not in st.session_state:
    st.session_state.final_netlist_generated = False

# Main content
st.title("⏚ CircuitVision")

# Show SAM2 status
if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2:
    if not st.session_state.get('model_load_toast_shown', False):
        st.toast("✅ Model loaded successfully")
        st.session_state.model_load_toast_shown = True
else:
    # This warning is appropriate to show if the condition persists (e.g. SAM2 files missing)
    st.warning("⚠️ Model loading failed")

# File upload section
uploaded_file = st.file_uploader(
    "Drag and drop your circuit diagram here",
    type=['png', 'jpg', 'jpeg'],
    help="For best results, use a clear image with good contrast"
)

if uploaded_file is not None:
    # Generate a consistent image path for the uploaded file
    image_path = os.path.join(UPLOAD_DIR, f'1.{uploaded_file.name.split(".")[-1]}')
    
    # Only reset results if a new file is uploaded
    if st.session_state.previous_upload_name != uploaded_file.name:
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
            'detailed_timings': {}, # Added for detailed timings
            'image_path': image_path  # Store the image path in session state
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
            pil_image.save(image_path, format=pil_image.format, exif=exif_bytes)
            logger.info(f"Saved image {image_path} with EXIF data ({len(exif_bytes)} bytes).")
        else:
            pil_image.save(image_path, format=pil_image.format)
            logger.info(f"Saved image {image_path} without EXIF data (EXIF not found in PIL image after processing).")
        
        # Update previous upload name
        st.session_state.previous_upload_name = uploaded_file.name
        
        # Automatically trigger analysis
        st.session_state.start_analysis_triggered = True
        st.session_state.analysis_in_progress = True
        st.rerun()  # Force immediate rerun to start analysis
    else:
        # If it's the same file, just ensure we have the image loaded
        if st.session_state.active_results['original_image'] is None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.active_results['original_image'] = image
    
    # Analysis logic, executed if triggered
    if st.session_state.get('start_analysis_triggered', False):
        st.session_state.start_analysis_triggered = False # Consume the trigger
        try:
            # analysis_in_progress is already True. 
            
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
                
                # Step 1: Detect Components
                step_start_time = time.time()
                if st.session_state.active_results['original_image'] is not None:
                    logger.debug("Step 1: Detecting components...")
                    # Get raw bounding boxes
                    raw_bboxes = analyzer.bboxes(st.session_state.active_results['original_image'])
                    logger.debug(f"Detected {len(raw_bboxes)} raw bounding boxes")
                    
                    # Apply Non-Maximum Suppression
                    bboxes = non_max_suppression_by_confidence(raw_bboxes, iou_threshold=0.6)
                    logger.debug(f"After NMS: {len(bboxes)} bounding boxes")
                    st.session_state.active_results['bboxes'] = bboxes
                    
                    # Create annotated image using the utility function
                    annotated_image = create_annotated_image(
                        st.session_state.active_results['original_image'], 
                        bboxes
                    )
                    st.session_state.active_results['annotated_image'] = annotated_image
                    
                    # Parse and display component counts using the utility function
                    component_stats = calculate_component_stats(bboxes)
                    logger.debug(f"Component statistics: {len(component_stats)} unique component types identified")
                    st.session_state.active_results['component_stats'] = component_stats
                else:
                    logger.error("No image found in session state")
                    raise ValueError("No image found. Please upload an image and try again.")
                detailed_timings['Component Detection'] = time.time() - step_start_time
                
                # Step 2: Node Analysis
                step_start_time = time.time()
                if st.session_state.active_results['bboxes'] is not None:
                    try:
                        logger.debug("Step 2: Analyzing node connections...")
                        nodes, emptied_mask, enhanced, contour_image, final_visualization = analyzer.get_node_connections(
                            st.session_state.active_results['original_image'], 
                            st.session_state.active_results['bboxes']
                        )
                        
                        logger.debug(f"Node analysis completed: {len(nodes)} nodes identified")
                        st.session_state.active_results['nodes'] = nodes
                        st.session_state.active_results['node_visualization'] = final_visualization
                        st.session_state.active_results['node_mask'] = emptied_mask
                        st.session_state.active_results['enhanced_mask'] = enhanced
                        st.session_state.active_results['contour_image'] = contour_image
                        
                        # Get SAM2 output if available
                        if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2 and hasattr(analyzer, 'last_sam2_output'):
                            logger.debug("SAM2 output available and stored")
                            st.session_state.active_results['sam2_output'] = analyzer.last_sam2_output
                            
                    except Exception as e:
                        error_msg = f"Error during node analysis: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
                        # Don't stop execution on node analysis error, but mark in log
                        logger.warning("Continuing execution despite node analysis error")
                else:
                    logger.error("No bounding boxes found for node analysis")
                    st.error("No components were detected. Please try a different image with clearer circuit elements.")
                detailed_timings['Node Analysis'] = time.time() - step_start_time
                
                # Step 3: Generate Netlist
                step_start_time = time.time()
                if st.session_state.active_results['nodes'] is not None:
                    try:
                        logger.debug("Step 3: Generating initial netlist...")
                        # Initial Netlist
                        valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.active_results['nodes'])
                        valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                        logger.debug(f"Generated initial netlist with {len(valueless_netlist)} components")
                        
                        # Store initial netlist
                        st.session_state.active_results['netlist'] = valueless_netlist
                        st.session_state.active_results['valueless_netlist_text'] = valueless_netlist_text
                        st.session_state.active_results['netlist_text'] = valueless_netlist_text  # Use valueless as initial netlist text
                        
                        # Prepare for final netlist generation
                        logger.debug("Enumerating components for later Gemini labeling...")
                        enum_img, bbox_ids = analyzer.enumerate_components(st.session_state.active_results['original_image'], st.session_state.active_results['bboxes'])
                        # Convert BGR to RGB for display and Gemini
                        if enum_img is not None:
                            enum_img = cv2.cvtColor(enum_img, cv2.COLOR_BGR2RGB)
                        st.session_state.active_results['enum_img'] = enum_img
                        st.session_state.active_results['bbox_ids'] = bbox_ids  # Store for later use
                        
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
                
                detailed_timings['Netlist Generation'] = time.time() - step_start_time
                
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
                img = Image.open(image_path)
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
                    exif = format_exif_data(image_path)
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
                
            else:
                st.info("Run analysis to see component detection")
        
        # Step 2: Node Analysis
        if st.session_state.active_results['contour_image'] is not None or st.session_state.active_results['sam2_output'] is not None:
            st.markdown("## 🔗 Node Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Contours")
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
                    st.image(st.session_state.active_results['contour_image'], caption="Final Node Connections", use_container_width=True)
        
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
                    valueless_netlist = st.session_state.active_results['netlist']
                    enum_img = st.session_state.active_results['enum_img']
                    bbox_ids = st.session_state.active_results['bbox_ids']
                    
                    # Create deep copy for the final netlist
                    netlist = deepcopy(valueless_netlist)
                    
                    # Call Gemini for component labeling
                    try:
                        logger.debug("Calling Gemini for component labeling...")
                        gemini_info = gemini_labels_openrouter(enum_img)
                        logger.debug(f"Received information for {len(gemini_info) if gemini_info else 0} components from Gemini")
                        logger.info(f"GEMINI BARE OUTPUT (gemini_info): {gemini_info}")
                        analyzer.fix_netlist(netlist, gemini_info, bbox_ids)
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
            
            # Display netlists in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Initial Netlist")
                if 'valueless_netlist_text' in st.session_state.active_results:
                    st.code(st.session_state.active_results['valueless_netlist_text'], language="python")
            
            with col2:
                st.markdown("### Final Netlist")
                if st.session_state.final_netlist_generated:
                    st.code(st.session_state.active_results['netlist_text'], language="python")
                else:
                    st.info("Click 'Get Final Netlist' to generate the final netlist with component values")
            
            # Show Gemini input in a dropdown
            with st.expander("🔍 Debug Gemini Input"):
                if 'enum_img' in st.session_state.active_results:
                    st.image(st.session_state.active_results['enum_img'], caption="Image sent to Gemini")
        
        # Step 4: SPICE Analysis - keep as is
        if st.session_state.active_results.get('netlist_text') is not None:
            st.markdown("## ⚡ SPICE Analysis")
            if st.button("Run SPICE Analysis"):
                try:
                    net_text = '.title detected_circuit\n' + st.session_state.active_results['netlist_text']
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
                    error_msg = f"❌ SPICE Analysis Error: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.info("💡 Tip: Check if all component values are properly detected and the circuit is properly connected.")