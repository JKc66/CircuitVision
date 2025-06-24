import streamlit as st
import os
import logging
import torch
import multiprocessing
import time
import re
from PIL import Image
import base64

from src.utils import (
            create_annotated_image,
            calculate_component_stats,
            load_css,
            format_exif_data
            )
from src.circuit_analyzer import CircuitAnalyzer
from src.analysis_pipeline import (
    process_new_upload,
    run_initial_detection,
    run_segmentation_and_cropping,
    run_terminal_reclassification,
    run_llama_enrichment,
    run_node_analysis,
    run_initial_netlist_generation,
    log_analysis_summary,
    handle_final_netlist_generation
)
from src.spice_simulator import (
    perform_dc_spice_analysis,
    perform_ac_spice_analysis
)



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

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("groq._base_client").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

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


# Load custom CSS
load_css("static/css/main.css")

def get_image_as_base64(path):
    """Reads an image file and returns it as a base64 encoded string."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        logger.error(f"Image file not found at path: {path}")
        return None

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

# Initialize session state for analyzer loading and toast
if 'circuit_analyzer_object_storage' not in st.session_state:
    st.session_state.circuit_analyzer_object_storage = None
if 'circuit_analyzer_loaded_flag' not in st.session_state:
    st.session_state.circuit_analyzer_loaded_flag = False
if 'model_load_toast_shown' not in st.session_state: # Keep existing toast flag
    st.session_state.model_load_toast_shown = False

# Initialize the circuit analyzer with error handling
@st.cache_resource(show_spinner=False) # MODIFIED: Disable default spinner
def load_circuit_analyzer():
    try:
        analyzer_obj = CircuitAnalyzer( # Renamed internal variable
            yolo_path=str(YOLO_MODEL_PATH),
            sam2_config_path=str(SAM2_CONFIG_PATH),
            sam2_base_checkpoint_path=str(SAM2_BASE_CHECKPOINT_PATH),
            sam2_finetuned_checkpoint_path=str(SAM2_FINETUNED_CHECKPOINT_PATH),
            use_sam2=use_sam2_feature, # use_sam2_feature is globally determined before this
            debug=True
        )
        return analyzer_obj
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        import traceback
        trace_msg = f"Traceback: {traceback.format_exc()}"
        logger.error(trace_msg)
        st.error(error_msg) # Display error in Streamlit UI
        st.error(trace_msg)
        return None

analyzer = None # Initialize analyzer variable

if not st.session_state.circuit_analyzer_loaded_flag:
    loader_placeholder = st.empty()
    loader_html = """
    <div class='initial-loader-container'>
        <div class='initial-loader-content'>
            <div class='circuit-model-loader'> 
                <div></div>
                <div></div>
                <div></div>
            </div>
            <p style='margin-top: 10px;'>Loading Circuit Analyzer models, please wait...</p>
        </div>
    </div>
    """
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
    
    analyzer_instance = load_circuit_analyzer() # Call the cached function
    
    loader_placeholder.empty() # Clear the custom loader

    if analyzer_instance is None:
        # Error message is already shown by load_circuit_analyzer via st.error
        logger.error("Circuit analyzer failed to initialize (custom loader). Stopping application.")
        st.stop() # Stop the app if model loading failed
    else:
        analyzer = analyzer_instance
        st.session_state.circuit_analyzer_object_storage = analyzer_instance
        st.session_state.circuit_analyzer_loaded_flag = True
        logger.info("Circuit analyzer initialized successfully (custom loader). Application ready.")
        
        # Mark that models have been loaded (no toast shown anymore)
        st.session_state.model_load_toast_shown = True
else:
    analyzer = st.session_state.circuit_analyzer_object_storage

# Final check: if analyzer is still None, something went wrong.
# This should ideally be caught by the st.stop() above.
if analyzer is None:
    st.error("FATAL: Circuit Analyzer is not available. Application cannot continue.", icon="🚨")
    logger.critical("FATAL: Analyzer object is None after loading logic. Forcing stop.")
    st.stop()

logger.info("--"*40) 
logger.info("--"*40)

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

# Get CPU info
cpu_count = multiprocessing.cpu_count()

# Display badges
if analyzer is not None and st.session_state.get('circuit_analyzer_loaded_flag', False):
    if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2:
        model_status_class = "success"
        model_status_text = "Models Ready (SAM2 Active)"
    else:
        model_status_class = "warning"
        if use_sam2_feature:
            model_status_text = "Models Ready (SAM2 Limited)"
        else:
            model_status_text = "Models Ready (No SAM2)"

    # Display badges and logo in columns
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start;">
            <div class="model-status-badge {model_status_class}">
                <div class="status-indicator {model_status_class}"></div>
                {model_status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        logo_path = os.path.join(BASE_DIR, "static/images/CircuitVision-nobg_dark.png")
        logo_base64 = get_image_as_base64(logo_path)
        if logo_base64:
            st.markdown(f"""
                <div class="logo-container">
                    <img src="data:image/png;base64,{logo_base64}" class="main-logo">
                </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback to the original method if base64 conversion fails
            st.image(logo_path, use_container_width=True)
    with col3:
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end;">
            <div class="cpu-info-badge">
                <div class="blinking-light"></div>
                Running on CPU ({cpu_count} cores)
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # If analyzer is not loaded yet, just show CPU badge on the right
    _, col2 = st.columns([3, 1])
    with col2:
        st.markdown(f"""
        <div class="cpu-info-badge">
            <div class="blinking-light"></div>
            Running on CPU ({cpu_count} cores)
        </div>
        """, unsafe_allow_html=True)


# Main content  
st.image(os.path.join(BASE_DIR, "static/images/sdp_banner_black.png"), use_container_width=True)

# File upload section
file_upload_container = st.container()
with file_upload_container:
    uploaded_file = st.file_uploader(
        "Read the help tooltip (?)",
        type=['png', 'jpg', 'jpeg'],
        help="""
        - Supports AC/DC linear circuits.
        - Dependent components are not yet supported.
        """
    )

# Process uploaded file
if uploaded_file is not None:    
    # Only reset results if a new file is uploaded
    if st.session_state.get('previous_upload_name') != uploaded_file.name:
        st.session_state.previous_upload_name = uploaded_file.name
        process_new_upload(uploaded_file, UPLOAD_DIR, logger)
        
# Analysis logic, executed if triggered
if st.session_state.get('start_analysis_triggered', False):
    st.session_state.start_analysis_triggered = False # Consume the trigger
    
    # Ensure an image is loaded and ready for analysis
    if st.session_state.active_results.get('original_image') is None or \
       st.session_state.active_results.get('image_path') is None:
        logger.warning("Analysis triggered, but required image data (original_image or image_path) is missing in active_results.")
        st.session_state.analysis_in_progress = False 
    else:
        try:
            # analysis_in_progress is already True if set by the trigger.
            
            # Create a placeholder for the custom loader and text
            loader_placeholder = st.empty()

            with loader_placeholder.container():
                st.markdown("""
                    <div class="loader-popup-overlay">
                        <div class="loader-popup-content">
                            <div class="analysis-loader"></div>
                            <p class="loader-popup-text" style="text-align: center; margin-top: 15px; font-size: 1.1em; color: var(--text-color);">Analyzing... 15 - 20 seconds</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Start timing the analysis
                overall_start_time = time.time()
                detailed_timings = {}
                logger.info("Starting complete circuit analysis...")
                
                # Step 1: Initial Component Detection
                # This step runs YOLO on the original image to get a raw list of components.
                try:
                    initial_bboxes = run_initial_detection(analyzer, st.session_state.active_results, detailed_timings, logger)
                    if initial_bboxes is None:
                        # Handle error case if run_initial_detection indicates a critical failure
                        st.error("Error: Could not perform initial component detection for cropping.")
                        st.session_state.analysis_in_progress = False
                        loader_placeholder.empty()
                        st.stop()
                except ValueError as ve:
                    st.error(str(ve))
                    st.session_state.analysis_in_progress = False
                    loader_placeholder.empty()
                    st.stop()

                # Step 2: Image Cropping and Segmentation
                # This step crops the image based on the location of detected components to focus on the circuit,
                # then runs SAM2 to segment the wires and connection paths on the (potentially) cropped image.
                image_for_analysis, bboxes_for_analysis, cropped_sam_mask_for_nodes = run_segmentation_and_cropping(analyzer, st.session_state.active_results, detailed_timings, logger)

                # Step 3: Terminal Reclassification on Cropped Image
                # On the newly cropped image, we re-evaluate components labeled as 'terminal'.
                # Based on their connectivity, they might be reclassified (e.g., to a voltage source).
                run_terminal_reclassification(analyzer, image_for_analysis, bboxes_for_analysis, detailed_timings, logger)

                # Step 4: LLaMA Enrichment on Cropped Image
                # Now that we have a cropped and reclassified view, we run LLaMA analysis on the
                # component images for better accuracy in determining semantic direction.
                run_llama_enrichment(analyzer, image_for_analysis, bboxes_for_analysis, detailed_timings, logger)

                # Step 5: Create Annotated Display Image
                # An annotated version of the (potentially cropped) image is created for display.
                annotated_display_image = create_annotated_image(
                    image_for_analysis, 
                    bboxes_for_analysis 
                )
                st.session_state.active_results['annotated_image'] = annotated_display_image
                
                # Calculate and store component statistics from the final set of bboxes
                component_stats = calculate_component_stats(bboxes_for_analysis)
                st.session_state.active_results['component_stats'] = component_stats

                # Step 6: Node Analysis
                # This step uses the SAM2 mask and adjusted component bboxes to identify connection nodes.
                nodes = run_node_analysis(analyzer, image_for_analysis, cropped_sam_mask_for_nodes, bboxes_for_analysis, st.session_state.active_results, detailed_timings, logger)
                
                # Step 7: Initial Netlist Generation
                # An initial netlist (without final component values) is generated from the identified nodes.
                valueless_netlist = run_initial_netlist_generation(analyzer, nodes, image_for_analysis, bboxes_for_analysis, st.session_state.active_results, detailed_timings, logger)

                # Log analysis summary
                log_analysis_summary(st.session_state.active_results, logger, log_level)
                
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

    # Create tabs for the results
    tab_titles = [
        "🔎 Overview",
        "🔗 Node Analysis",
        "📝 Netlist",
        "⚡ Simulation"
    ]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        st.markdown("## 📈 Analysis Results")
        
        # Display analysis time if available
        if 'elapsed_time' in st.session_state.active_results:
            expander_label = f"⏱️ Analysis Time: `{st.session_state.active_results['elapsed_time']:.2f}s` (Expand for step details)"
            st.markdown("<div class='detailed-timings-expander'>", unsafe_allow_html=True)
            with st.expander(expander_label):
                detailed_timings = st.session_state.active_results.get('detailed_timings')
                if detailed_timings:
                    # Find min and max durations for color scaling
                    durations = list(detailed_timings.values())
                    min_duration = min(durations) if durations else 0
                    max_duration = max(durations) if durations else 1

                    def get_color_for_duration(duration, min_d, max_d):
                        """Calculates an HSL color from green to red based on duration for a dark theme."""
                        if max_d == min_d:
                            normalized = 0.5
                        else:
                            normalized = (duration - min_d) / (max_d - min_d)
                        
                        # Hue for green is 120, hue for red is 0.
                        # We invert normalized value so 0 is green and 1 is red.
                        hue = 120 * (1 - normalized)
                        hue = max(0, min(hue, 120)) # Clamp hue
                        
                        # Use lower lightness for dark backgrounds
                        return f"hsl({hue}, 60%, 25%)"

                    # Build an HTML table with inline styles for row colors
                    table_html = """
                    <style>
                        .timing-table { width: 100%; border-collapse: collapse; color: var(--text-color); }
                        .timing-table th, .timing-table td { padding: 8px; text-align: left; border-bottom: 1px solid #444; }
                        .timing-table th { background-color: #2a2a2a; }
                    </style>
                    <table class='timing-table'>
                        <tr><th>Step</th><th>Duration (s)</th></tr>
                    """
                    
                    for step, duration in detailed_timings.items():
                        row_color = get_color_for_duration(duration, min_duration, max_duration)
                        table_html += f'<tr style="background-color: {row_color};"><td>{step}</td><td>{duration:.2f}</td></tr>'
                    
                    table_html += "</table>"
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.info("No detailed timing information available.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(st.session_state.active_results['original_image'], use_container_width=True)
            
            # Show image details as a dropdown
            with st.expander("🖼️ Image Details"):
                # Basic image properties that are always available
                img = Image.open(current_image_path) # Use current_image_path
                h, w = st.session_state.active_results['original_image'].shape[:2]
                st.markdown("### Basic Properties")
                st.markdown(f"- **Size**: `{w}x{h}` pixels")
                st.markdown(f"- **Format**: `{img.format}`")
                st.markdown(f"- **Mode**: `{img.mode}`")
                if 'uploaded_file_name' in st.session_state.active_results and st.session_state.active_results['uploaded_file_name'] is not None:
                    st.markdown(f"- **Name**: `{st.session_state.active_results['uploaded_file_name']}`")
                else:
                    st.markdown("- **Name**: `N/A`")
                
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
                        st.info("ℹ️ No EXIF metadata found. (Common for processed images/screenshots)")
                except Exception as e:
                    st.warning(f"Could not read EXIF data: {str(e)}")
                    logger.error(f"Error displaying EXIF data: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            # End of "Image Details" expander

        # New, separate expander for Cropping Debug Information
        with st.expander("✂️ Cropping Details"): # Updated title
            crop_info = st.session_state.active_results.get('crop_debug_info')
            if crop_info:
                if crop_info.get('crop_applied'):
                    st.success("✅ Cropping applied successfully.")
                else:
                    st.warning(f"⚠️ Cropping NOT applied. Reason: {crop_info.get('reason_for_no_crop', 'Unknown')}")
                
                orig_dims_tuple = crop_info.get('original_image_dims', ('N/A', 'N/A'))
                cropped_dims_tuple = crop_info.get('cropped_image_dims', ('N/A', 'N/A'))
                # Ensure they are tuples for consistent string representation
                orig_dims_str = f"({orig_dims_tuple[0]}, {orig_dims_tuple[1]})" if isinstance(orig_dims_tuple, tuple) and len(orig_dims_tuple) == 2 else str(orig_dims_tuple)
                cropped_dims_str = f"({cropped_dims_tuple[0]}, {cropped_dims_tuple[1]})" if isinstance(cropped_dims_tuple, tuple) and len(cropped_dims_tuple) == 2 else str(cropped_dims_tuple)

                st.markdown(f"**Dimensions (Original WxH → Cropped WxH):** `{orig_dims_str}` → `{cropped_dims_str}`")

                st.markdown("**Crop Definition:**")
                st.markdown(f"- SAM2 Extent Bbox: `{crop_info.get('defining_bbox', 'N/A')}`")
                st.markdown(f"- Padding Applied: `{crop_info.get('padding_value', 'N/A')}px`") # Added px for clarity
                st.markdown(f"- Initial Window (Post-Padding): `{crop_info.get('initial_calculated_window_before_text', 'N/A')}`")
                
                text_expanders = crop_info.get('text_bboxes_that_expanded_crop', [])
                if text_expanders:
                    st.markdown(f"**Text-based Expansion:** {len(text_expanders)} text region(s) expanded the crop area.")
                    # Using st.markdown with manual list formatting for consistent indentation
                    expansion_details = ""
                    for i, txt_bbox_info in enumerate(text_expanders):
                        expansion_details += f"  - Text Box {i+1}: UID=`{txt_bbox_info.get('uid')}`, Coords=`{txt_bbox_info.get('coords_text_box_abs')}`\n"
                    st.markdown(expansion_details)
                else:
                    st.markdown("**Text-based Expansion:** No text regions further expanded the crop area.")
                
                st.markdown(f"**Final Crop Window (xmin, ymin, xmax, ymax):** `{crop_info.get('final_crop_window_abs', 'N/A')}`")
                
            else:
                st.info("No cropping debug information available (crop may not have run or info not stored).")

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
                            # Render as an HTML table for custom styling
                            table_html = "<div class='component-stats-table'><table>"
                            table_html += "<thead><tr><th>Component</th><th>N</th><th>Avg Conf</th></tr></thead>"
                            table_html += "<tbody>"
                            for item in sorted(stats_data, key=lambda x: x['Component']):
                                table_html += f"<tr><td>{item['Component']}</td><td>{item['N']}</td><td>{item['Avg Conf']}</td></tr>"
                            table_html += "</tbody></table></div>"
                            st.markdown(table_html, unsafe_allow_html=True)
                
                # Expander for Initial YOLO Detections (on original image)
                with st.expander("🔍 Debug: Original Image"):
                    if 'bboxes_orig_coords_nms' in st.session_state.active_results and \
                       st.session_state.active_results['bboxes_orig_coords_nms'] is not None and \
                       'original_image' in st.session_state.active_results and \
                       st.session_state.active_results['original_image'] is not None:
                        
                        # Create an annotated image using the *original* image and *original coordinates* bboxes
                        # Ensure create_annotated_image can handle the bboxes_orig_coords_nms format
                        initial_yolo_annotated_img = create_annotated_image(
                            st.session_state.active_results['original_image'],
                            st.session_state.active_results['bboxes_orig_coords_nms']
                        )
                        st.image(initial_yolo_annotated_img, caption="Initial YOLO detections on original image before any cropping (except NMS)", use_container_width=True)
                    else:
                        st.info("Initial YOLO detection data or original image not available in session state.")

                # Expander for LLaMA Stage 1 (Direction) Debug Output
                with st.expander("↗️ Debug: Source Directions"):
                    # Check prerequisites first
                    if 'bboxes' not in st.session_state.active_results or not st.session_state.active_results.get('bboxes'):
                        st.info("Processed component bounding boxes ('bboxes') not available in session state. Run analysis first.")
                    elif not hasattr(analyzer, 'last_vlm_input_images'):
                        st.info("VLM input images attribute ('last_vlm_input_images') not found on analyzer. Ensure CircuitAnalyzer initializes this attribute in debug mode.")
                    elif not analyzer.last_vlm_input_images: # This means the dict exists but is empty
                        st.info("VLM debug: No input images were stored by the analyzer for VLM processing. Check component types or analyzer logic if images were expected.")
                    else:
                        # This is the success case: bboxes exist, attribute exists, and dict is not empty
                        st.success("✅ Directions found")
                        
                        found_at_least_one_image_to_display = False
                        for comp_bbox in st.session_state.active_results['bboxes']:
                            component_uid = comp_bbox.get('persistent_uid')
                            # Only include components for which VLM analysis was attempted, an image is available, and UID exists
                            if 'semantic_direction' in comp_bbox and comp_bbox['semantic_direction'] is not None and \
                               component_uid and component_uid in analyzer.last_vlm_input_images:
                                
                                vlm_input_image = analyzer.last_vlm_input_images[component_uid]
                                
                                yolo_class = comp_bbox.get('class', 'N/A')
                                semantic_direction = comp_bbox['semantic_direction']
                                semantic_reason = comp_bbox.get('semantic_reason', 'N/A')
                                
                                interpreted_type = yolo_class # Default to YOLO class
                                if yolo_class in analyzer.voltage_classes_names and semantic_reason == "ARROW":
                                    interpreted_type = "current.ac" if ".ac" in yolo_class else "current.dc"
                                elif yolo_class in analyzer.current_source_classes_names and semantic_reason == "SIGN":
                                    interpreted_type = "voltage.ac" if ".ac" in yolo_class else "voltage.dc"
                                
                                output_line = (
                                    f"{yolo_class} `{semantic_direction}`;`{semantic_reason}`→`{interpreted_type}`"
                                )
                                
                                st.image(vlm_input_image, width=100)
                                st.markdown(output_line)
                                found_at_least_one_image_to_display = True
                        
                        if not found_at_least_one_image_to_display:
                            st.info("No components found with matching UIDs and semantic direction data.")
            
            else:
                st.info("Run analysis to see component detection")

    with tab2:
        # Step 2: Node Analysis
        if st.session_state.active_results['contour_image'] is not None or st.session_state.active_results['sam2_output'] is not None:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Contours")
                if st.session_state.active_results['contour_image'] is not None:
                    st.image(st.session_state.active_results['contour_image'], use_container_width=True)
            
            with col2:
                st.markdown("### SAM2 Segmentation")
                if st.session_state.active_results['sam2_output'] is not None:
                    st.image(st.session_state.active_results['sam2_output'], use_container_width=True)
            
            # Show additional debug images in a dropdown
            with st.expander("🖼️ Debug Images"):
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    if st.session_state.active_results['node_mask'] is not None:
                        st.image(st.session_state.active_results['node_mask'], caption="`Emptied Mask`", use_container_width=True)
                
                with debug_col2:
                    if st.session_state.active_results['enhanced_mask'] is not None:
                        st.image(st.session_state.active_results['enhanced_mask'], caption="`Enhanced Mask`", use_container_width=True)
                
                
                if st.session_state.active_results.get('connection_points_image') is not None:
                    st.image(st.session_state.active_results['connection_points_image'], caption="`Connection Points`", use_container_width=True)

                if st.session_state.active_results['node_visualization'] is not None:
                    st.image(st.session_state.active_results['node_visualization'], caption="`Final Node Connections`", use_container_width=True)
        else:
            st.info("Node analysis results are not available. This may be because SAM2 is disabled or the analysis step failed.")

    with tab3:
        # Step 3: Netlist
        if st.session_state.active_results.get('netlist_text') is not None:
            # Modified layout to align column sizes better
            
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
                handle_final_netlist_generation(analyzer, st.session_state.active_results, logger)
            
            # Main netlist display: Initial (VLM) and Final side-by-side
            col1_main_netlist, col2_main_netlist = st.columns(2)
            with col1_main_netlist:
                st.markdown("### Initial Netlist") # (This is with VLM directions)
                if 'valueless_netlist_text' in st.session_state.active_results:
                    st.code(st.session_state.active_results['valueless_netlist_text'], language="verilog")
                else:
                    st.info("Initial netlist not available.")

            with col2_main_netlist:
                st.markdown("### Final Netlist") # (After VLM Stage 2)
                if st.session_state.final_netlist_generated and 'netlist_text' in st.session_state.active_results:
                    # MODIFIED: Use st.code for displaying the final netlist
                    st.code(st.session_state.active_results['netlist_text'], language="verilog")
                elif not st.session_state.final_netlist_generated:
                    st.info("Click 'Get Final Netlist' button above to generate.")
                else:
                    st.info("Final netlist not available.")
            
            # Dropdown 1: Image Sent to Gemini/VLM
            with st.expander("🔍 Debug: VLM"):
                st.markdown("### Image Sent to VLM")
                if 'enum_img' in st.session_state.active_results and st.session_state.active_results['enum_img'] is not None:
                    st.image(st.session_state.active_results['enum_img'], caption="`Image sent for VLM`")
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
                with st.expander("⇄ Initial Netlist Differences"):
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
                            st.markdown(f"**Line {i+1}:** `{line_l_content}` → `{line_nl_content}`")
                    
                    if not any_line_diff_shown:
                        st.markdown("The netlists are marked as different, but no specific line-by-line variances were rendered. This could be due to subtle differences like trailing whitespace. The initial netlist (with VLM) is shown above.")
        else:
            st.info("Netlist not available. Please run the analysis.")

    with tab4:
        # Only show SPICE analysis options AFTER the final netlist is generated
        if st.session_state.get('final_netlist_generated', False):
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
            # Auto-detected Analysis Type
            st.session_state.analysis_mode = determined_analysis_mode
            
            st.markdown(f"**Analysis Type:** `{st.session_state.analysis_mode}`")

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
                st.session_state.ac_frequency = ac_frequency_hz # Update session state with the input value


            if st.button("Run SPICE Analysis", type="secondary"):
                try: # General try for the button action
                    if st.session_state.analysis_mode == "DC (.op)":
                        perform_dc_spice_analysis(st.session_state.editable_netlist_content, logger)

                    elif st.session_state.analysis_mode == "AC (.ac)":
                        # Pass the potentially updated ac_frequency_hz from the number_input
                        perform_ac_spice_analysis(st.session_state.active_results, analyzer, st.session_state.ac_frequency, logger)
                
                except Exception as e: # This except block is for the general try block associated with the st.button action
                    error_msg = f"❌ SPICE Analysis Main Error: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                    st.info("💡 Tip: An unexpected error occurred during SPICE analysis setup.")
        # If final_netlist_generated is False, but an initial netlist (valueless) exists
        elif st.session_state.active_results.get('netlist_text') is not None:
            st.info("Please click 'Get Final Netlist' in the '📝 Netlist' tab to enable SPICE analysis.")
        else:
            st.info("SPICE analysis requires a generated netlist. Please run the analysis first.")

# Add footer here
footer_html = """
<div class='custom-footer'>
  <h3 class='custom-footer-h3'>Project & Authors</h3>
  <div class='footer-links-container'>
    <a href='https://www.linkedin.com/in/mah-sam/' target='_blank' rel='noopener noreferrer' class='footer-link'>
      <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="footer-icon linkedin-icon"><title>LinkedIn</title><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>
      <span>Mahmoud Sameh</span>
    </a>
    <a href='https://www.linkedin.com/in/jawadk-c66/' target='_blank' rel='noopener noreferrer' class='footer-link'>
      <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="footer-icon linkedin-icon"><title>LinkedIn</title><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.225 0z"/></svg>
      <span>Jawad K</span>
    </a>
    <a href='https://github.com/JKc66/CircuitVision' target='_blank' rel='noopener noreferrer' class='footer-link'>
      <svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" class="footer-icon github-icon"><title>GitHub</title><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>
      <span>CircuitVision</span>
    </a>
  </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)