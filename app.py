import torch
import numpy as np
import cv2
import shutil
import streamlit as st
import logging
import os
import time
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
from pathlib import Path

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

torch.classes.__path__ = []

# Set page config and custom styles
st.set_page_config(
    page_title="CircuitVision",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
load_css("static/css/main.css")

# Set up base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'static/uploads'
YOLO_MODEL_PATH = BASE_DIR / 'models/YOLO/best_large_model_yolo.pt'
# Add SAM2 paths - use absolute paths for SAM2 as it requires specific path handling
SAM2_CONFIG_PATH = Path(r"D:\SDP_demo\models\configs\sam2.1_hiera_l.yaml")
SAM2_BASE_CHECKPOINT_PATH = Path(r"D:\SDP_demo\models\SAM2\sam2.1_hiera_large.pt")  # Base model
SAM2_FINETUNED_CHECKPOINT_PATH = Path(r"D:\SDP_demo\models\SAM2\best_miou_model_SAM_latest.pth")  # Fine-tuned weights

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"YOLO model path: {YOLO_MODEL_PATH}")

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'models/SAM2').mkdir(parents=True, exist_ok=True)  # Ensure models/SAM2 exists
logger.info(f"Created necessary directories: {UPLOAD_DIR}")

# Check if SAM2 model files exist
if not SAM2_CONFIG_PATH.exists() or not SAM2_BASE_CHECKPOINT_PATH.exists() or not SAM2_FINETUNED_CHECKPOINT_PATH.exists():
    missing_files = []
    if not SAM2_CONFIG_PATH.exists():
        missing_files.append(f"Config: {SAM2_CONFIG_PATH}")
    if not SAM2_BASE_CHECKPOINT_PATH.exists():
        missing_files.append(f"Base Checkpoint: {SAM2_BASE_CHECKPOINT_PATH}")
    if not SAM2_FINETUNED_CHECKPOINT_PATH.exists():
        missing_files.append(f"Fine-tuned Checkpoint: {SAM2_FINETUNED_CHECKPOINT_PATH}")
    
    logger.warning(f"One or more SAM2 model files not found:\n" + "\n".join(missing_files) + "\nSAM2 features will be disabled.")
    st.warning(f"One or more SAM2 model files not found:\n" + "\n".join(missing_files) + "\nSAM2 features will be disabled.")
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

# Main content
st.title("CircuitVision")

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
            'detailed_timings': {} # Added for detailed timings
        }
        
        # Store original image
        st.session_state.active_results['original_image'] = image
        
        # Clear and save files
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / f'1.{uploaded_file.name.split(".")[-1]}'
        save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        cv2.imwrite(str(image_path), save_image)
        
        # Update previous upload name
        st.session_state.previous_upload_name = uploaded_file.name
    else:
        # If it's the same file, just ensure we have the image loaded
        if st.session_state.active_results['original_image'] is None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.active_results['original_image'] = image
    
    # Add a prominent "Start Analysis" button
    if st.button("⚡ Start Analysis", use_container_width=True, type="primary"):
        
        # Create a placeholder for the custom loader and text
        loader_placeholder = st.empty()

        with loader_placeholder.container():
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
                    <div class="loader"></div>
                    <p style="text-align: center; margin-top: 15px; font-size: 1.1em; color: #333;">Running complete circuit analysis...</p>
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
            detailed_timings['Node Analysis'] = time.time() - step_start_time
            
            # Step 3: Generate Netlist
            step_start_time = time.time()
            if st.session_state.active_results['nodes'] is not None:
                logger.debug("Step 3: Generating netlist...")
                # Initial Netlist
                valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.active_results['nodes'])
                valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                logger.debug(f"Generated initial netlist with {len(valueless_netlist)} components")
                
                # Final Netlist
                netlist = deepcopy(valueless_netlist)
                logger.debug("Enumerating components for Gemini labeling...")
                enum_img, bbox_ids = analyzer.enumerate_components(st.session_state.active_results['original_image'], st.session_state.active_results['bboxes'])
                logger.debug("Calling Gemini for component labeling...")
                gemini_info = gemini_labels_openrouter(enum_img)
                logger.debug(f"Received information for {len(gemini_info) if gemini_info else 0} components from Gemini")
                analyzer.fix_netlist(netlist, gemini_info, bbox_ids)
                
                # Debug: Log the state of netlist lines before stringifying
                if logger.isEnabledFor(logging.DEBUG):
                    for i, ln_debug in enumerate(netlist):
                        logger.debug(f"App.py netlist line {i} before stringify: {ln_debug}")
                        
                netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])
                logger.debug("Final netlist generation complete")
                
                st.session_state.active_results['netlist'] = netlist
                st.session_state.active_results['netlist_text'] = netlist_text
                st.session_state.active_results['valueless_netlist_text'] = valueless_netlist_text
                st.session_state.active_results['enum_img'] = enum_img
                
                # Log analysis summary
                if log_level in ['DEBUG', 'INFO']:
                    component_counts = {}
                    for line in netlist:
                        comp_type = line['class']
                        if comp_type not in component_counts:
                            component_counts[comp_type] = 0
                        component_counts[comp_type] += 1
                    
                    logger.info("Analysis results summary:")
                    logger.info(f"- Image: {st.session_state.active_results.get('uploaded_file_name', 'Unknown')}")
                    logger.info(f"- Total components detected: {len(netlist)}")
                    for comp_type, count in component_counts.items():
                        logger.info(f"  - {comp_type}: {count}")
                    logger.info(f"- Total nodes: {len(st.session_state.active_results['nodes'])}")
            
            detailed_timings['Netlist Generation'] = time.time() - step_start_time
            
            # End timing and log the duration
            overall_end_time = time.time()
            elapsed_time = overall_end_time - overall_start_time
            logger.info(f"Circuit analysis completed in {elapsed_time:.2f} seconds")
            
            # Store elapsed time and detailed timings in active results
            st.session_state.active_results['elapsed_time'] = elapsed_time
            st.session_state.active_results['detailed_timings'] = detailed_timings

            # Clear the custom loader and text
            loader_placeholder.empty()
    
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
                h, w = st.session_state.active_results['original_image'].shape[:2]
                st.markdown(f"- **Size**: {w}x{h} pixels")
                if 'uploaded_file_type' in st.session_state.active_results:
                    st.markdown(f"- **Format**: {st.session_state.active_results['uploaded_file_type']}")
                if 'uploaded_file_name' in st.session_state.active_results:
                    st.markdown(f"- **Name**: {st.session_state.active_results['uploaded_file_name']}")
        
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
                if st.session_state.active_results['contour_image'] is not None:
                    st.image(st.session_state.active_results['contour_image'], use_container_width=True)
            
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
                
                if st.session_state.active_results['node_visualization'] is not None:
                    st.image(st.session_state.active_results['node_visualization'], caption="Final Node Connections", use_container_width=True)
        
        # Step 3: Netlist
        if st.session_state.active_results.get('netlist_text') is not None:
            st.markdown("## 📝 Circuit Netlist")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Initial Netlist")
                if 'valueless_netlist_text' in st.session_state.active_results:
                    st.code(st.session_state.active_results['valueless_netlist_text'], language="python")
            
            with col2:
                st.markdown("### Final Netlist")
                st.code(st.session_state.active_results['netlist_text'], language="python")
            
            # Show Gemini input in a dropdown
            with st.expander("🔍 Debug Gemini Input"):
                if 'enum_img' in st.session_state.active_results:
                    st.image(st.session_state.active_results['enum_img'], caption="Image sent to Gemini", width=400)
        
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