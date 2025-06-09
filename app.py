import torch
import streamlit as st
import logging
import os
import time
import re
from PIL import Image
from src.utils import (
            create_annotated_image,
            calculate_component_stats,
            load_css,
            format_exif_data
            )
from src.circuit_analyzer import CircuitAnalyzer
from src.analysis_pipeline import (
    process_new_upload,
    run_initial_detection_and_enrichment,
    run_segmentation_and_cropping,
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
        
        # Show toast notification for model loading status
        if not st.session_state.model_load_toast_shown:
            if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2:
                st.toast("✅ Models loaded successfully (including SAM2).")
            else:
                # This condition implies models loaded, but SAM2 might be off or had an issue.
                # use_sam2_feature (checked during CircuitAnalyzer init) determines if SAM2 checkpoint files were found.
                if use_sam2_feature: # SAM2 files were found, but it's still not active on analyzer
                    st.toast("✅ Core models loaded. SAM2 initialized but may not be fully active.", icon="⚠️")
                else: # SAM2 files were not found initially
                    st.toast("✅ Core models loaded. SAM2 features disabled (model files not found).", icon="⚠️")
            st.session_state.model_load_toast_shown = True
else:
    analyzer = st.session_state.circuit_analyzer_object_storage

# Final check: if analyzer is still None, something went wrong.
# This should ideally be caught by the st.stop() above.
if analyzer is None:
    st.error("FATAL: Circuit Analyzer is not available. Application cannot continue.", icon="🚨")
    logger.critical("FATAL: Analyzer object is None after loading logic. Forcing stop.")
    st.stop()

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
                            <div class="loader"></div>
                            <p style="text-align: center; margin-top: 15px; font-size: 1.1em; color: #333;">please wait ...</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Start timing the analysis
                overall_start_time = time.time()
                detailed_timings = {}
                logger.info("Starting complete circuit analysis...")
                
                # # Step 1A: Initial Component Detection (YOLO on original image)
                # # Step 1AA: Enrich BBoxes with Semantic Directions from LLaMA (Groq)
                # The two steps above are now combined into run_initial_detection_and_enrichment
                try:
                    bboxes_after_enrichment = run_initial_detection_and_enrichment(analyzer, st.session_state.active_results, detailed_timings, logger)
                    if bboxes_after_enrichment is None:
                        # Handle error case if run_initial_detection_and_enrichment indicates a critical failure
                        # For example, if original_image was missing
                        st.error("Error: Could not perform initial component detection.")
                        st.session_state.analysis_in_progress = False
                        loader_placeholder.empty()
                        st.stop()
                    # bboxes_orig_coords_nms in st.session_state.active_results is updated by the function
                except ValueError as ve:
                    st.error(str(ve))
                    st.session_state.analysis_in_progress = False
                    loader_placeholder.empty()
                    st.stop()

                # # Step 1B: SAM2 Segmentation (on original image) & Get SAM2 Extent
                # # Step 1C: Crop based on SAM2 Extent
                # These steps are now combined into run_segmentation_and_cropping
                image_for_analysis, bboxes_for_analysis, cropped_sam_mask_for_nodes = run_segmentation_and_cropping(analyzer, st.session_state.active_results, detailed_timings, logger)

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

                # # Step 2: Node Analysis (using cropped SAM2 mask and adjusted bboxes)
                # This step is now in run_node_analysis
                nodes = run_node_analysis(analyzer, image_for_analysis, cropped_sam_mask_for_nodes, bboxes_for_analysis, st.session_state.active_results, detailed_timings, logger)
                # `nodes` variable now holds the result, and st.session_state.active_results['nodes'] is also updated within the function.

                # # Step 3: Generate Netlist
                # This step is now in run_initial_netlist_generation
                valueless_netlist = run_initial_netlist_generation(analyzer, nodes, image_for_analysis, bboxes_for_analysis, st.session_state.active_results, detailed_timings, logger)
                # `valueless_netlist` holds the result, and session state is updated within the function.

                # Log analysis summary (Moved definition earlier, called here)
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

    # Step 1: Image and Component Detection side by side
    st.markdown("## 📈 Analysis Results")
    
    # Display analysis time if available
    if 'elapsed_time' in st.session_state.active_results:
        expander_label = f"⏱️ Analysis Time: {st.session_state.active_results['elapsed_time']:.2f}s (Expand for step details)"
        st.markdown("<div class='detailed-timings-expander'>", unsafe_allow_html=True)
        with st.expander(expander_label):
            detailed_timings = st.session_state.active_results['detailed_timings']
            if detailed_timings:
                table_data = ["| Step | Duration (s) |", "| :--- | :----------- |"]
                for step, duration in detailed_timings.items():
                    table_data.append(f"| {step} | {duration:.2f} |")
                st.markdown("\n".join(table_data), unsafe_allow_html=True)
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
                st.markdown(f"- **Name**: `N/A`")
            
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
                        st.table(stats_data)
            
            # Expander for Initial YOLO Detections (on original image)
            with st.expander("🔍 Debug: Initial YOLO Detections (Original Image)"):
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
                    st.image(initial_yolo_annotated_img, caption="Initial YOLO detections on original image before any cropping or reclassification (except NMS)", use_container_width=True)
                else:
                    st.info("Initial YOLO detection data or original image not available in session state.")

            # Expander for LLaMA Stage 1 (Direction) Debug Output
            with st.expander("↗️ Debug: LLaMA Directions"):
                # Check prerequisites first
                if 'bboxes' not in st.session_state.active_results or not st.session_state.active_results.get('bboxes'):
                    st.info("Processed component bounding boxes ('bboxes') not available in session state. Run analysis first.")
                elif not hasattr(analyzer, 'last_llama_input_images'):
                    st.info("LLaMA input images attribute ('last_llama_input_images') not found on analyzer. Ensure CircuitAnalyzer initializes this attribute in debug mode.")
                elif not analyzer.last_llama_input_images: # This means the dict exists but is empty
                    st.info("LLaMA debug: No input images were stored by the analyzer for LLaMA processing. Check component types or analyzer logic if images were expected.")
                else:
                    # This is the success case: bboxes exist, attribute exists, and dict is not empty
                    st.success("✅")
                    
                    found_at_least_one_image_to_display = False
                    for comp_bbox in st.session_state.active_results['bboxes']:
                        component_uid = comp_bbox.get('persistent_uid')
                        # Only include components for which LLaMA analysis was attempted, an image is available, and UID exists
                        if 'semantic_direction' in comp_bbox and comp_bbox['semantic_direction'] is not None and \
                           component_uid and component_uid in analyzer.last_llama_input_images:
                            
                            llama_input_image = analyzer.last_llama_input_images[component_uid]
                            
                            yolo_class = comp_bbox.get('class', 'N/A')
                            semantic_direction = comp_bbox['semantic_direction']
                            semantic_reason = comp_bbox.get('semantic_reason', 'N/A')
                            
                            interpreted_type = yolo_class # Default to YOLO class
                            if yolo_class in analyzer.voltage_classes_names and semantic_reason == "ARROW":
                                interpreted_type = "current.ac" if ".ac" in yolo_class else "current.dc"
                            elif yolo_class in analyzer.current_source_classes_names and semantic_reason == "SIGN":
                                interpreted_type = "voltage.ac" if ".ac" in yolo_class else "voltage.dc"
                            
                            output_line = f"{yolo_class} `{semantic_direction}` ; `{semantic_reason}` &#8594; `{interpreted_type}`"
                            
                            st.image(llama_input_image, width=100)
                            st.markdown(output_line, unsafe_allow_html=True)
                            found_at_least_one_image_to_display = True
                    
                    if not found_at_least_one_image_to_display:
                        st.info("LLaMA input images are available in the analyzer, but no components in the current results have matching UIDs with stored images or the required semantic direction information to display their specific LLaMA input image.")
        
        else:
            st.info("Run analysis to see component detection")
    
    # Step 2: Node Analysis
    if st.session_state.active_results['contour_image'] is not None or st.session_state.active_results['sam2_output'] is not None:
        st.markdown("## 🔗 Node Analysis")
        
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
            handle_final_netlist_generation(analyzer, st.session_state.active_results, logger)
        
        # Main netlist display: Initial (LLaMA) and Final side-by-side
        col1_main_netlist, col2_main_netlist = st.columns(2)
        with col1_main_netlist:
            st.markdown("### Initial Netlist") # (This is with LLaMA directions)
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
                st.image(st.session_state.active_results['enum_img'], caption="`Image sent for VLM Stage 2 analysis`")
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
        st.markdown("## ⚡ SPICE Analysis")
        st.info("Please click 'Get Final Netlist' above to enable SPICE analysis and component value detection.")

# Add footer here
footer_html = """
<div class='custom-footer'>
  <hr class='custom-footer-hr'>
  <h3 class='custom-footer-h3'>Connect with Us</h3>
  <div class='footer-columns'>
    <div class='footer-column'>
      <a href='https://www.linkedin.com/in/mah-sam/' target='_blank' rel='noopener noreferrer'>
        <img src='https://img.shields.io/badge/LinkedIn-Mahmoud%20Sameh-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='Mahmoud Sameh LinkedIn'>
      </a>
    </div>
    <div class='footer-column'>
      <a href='https://www.linkedin.com/in/jawadk-c66/' target='_blank' rel='noopener noreferrer'>
        <img src='https://img.shields.io/badge/LinkedIn-Jawad%20K-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='Jawad K LinkedIn'>
      </a>
    </div>
    <div class='footer-column'>
      <a href='https://github.com/JKc66/CircuitVision' target='_blank' rel='noopener noreferrer'>
        <img src='https://img.shields.io/badge/GitHub-CircuitVision-181717?style=for-the-badge&logo=github&logoColor=white' alt='CircuitVision GitHub'>
      </a>
    </div>
  </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)