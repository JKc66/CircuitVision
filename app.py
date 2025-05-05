import torch
import numpy as np
import cv2
import shutil
import streamlit as st
from src.utills import (summarize_components,
                        gemini_labels,
                        non_max_suppression_by_confidence)
from src.circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Spice.Parser import SpiceParser
from pathlib import Path

torch.classes.__path__ = []

# Set page config and custom styles
st.set_page_config(
    page_title="Circuit Analyzer",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Set up base paths
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'static/uploads'
MODEL_PATH = BASE_DIR / 'models/models/YOLO/best_large_model_yolo.pt'

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the circuit analyzer with error handling
def load_circuit_analyzer():
    try:
        analyzer = CircuitAnalyzer(yolo_path=str(MODEL_PATH), debug=False)
        return analyzer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the circuit analyzer
analyzer = load_circuit_analyzer()

# Create containers for results
if 'results' not in st.session_state:
    st.session_state.results = {
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': None,
        'annotated_image': None,
        'component_stats': None,
        'node_visualization': None,
        'node_mask': None,
        'contour_image': None
    }

# Track previous upload to prevent unnecessary resets
if 'previous_upload_name' not in st.session_state:
    st.session_state.previous_upload_name = None

# Main content
st.title("Circuit Diagram Analysis Tool")
st.markdown("Upload your circuit diagram and analyze it step by step.")

# File upload section
uploaded_file = st.file_uploader(
    "Drag and drop your circuit diagram here",
    type=['png', 'jpg', 'jpeg'],
    help="For best results, use a clear image with good contrast"
)

if uploaded_file is not None:
    # Only reset results if a new file is uploaded
    if st.session_state.previous_upload_name != uploaded_file.name:
        # Convert image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
        
        # Clear results when new image is uploaded
        st.session_state.results = {
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
            'contour_image': None
        }
        
        # Store original image
        st.session_state.results['original_image'] = image
        
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
        if st.session_state.results['original_image'] is None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.session_state.results['original_image'] = image
    
    # Step 1: Image Analysis
    st.markdown("## Step 1: 📸 Image Analysis")
    analyze_container = st.container()

    # Always show the original image if it exists
    if st.session_state.results['original_image'] is not None:
        with analyze_container:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(st.session_state.results['original_image'], caption="Original Image", use_container_width=True)
            with col2:
                st.markdown("### Image Details")
                h, w = st.session_state.results['original_image'].shape[:2]
                st.markdown(f"- **Size**: {w}x{h} pixels")
                if 'uploaded_file_type' in st.session_state.results:
                    st.markdown(f"- **Format**: {st.session_state.results['uploaded_file_type']}")
                if 'uploaded_file_name' in st.session_state.results:
                    st.markdown(f"- **Name**: {st.session_state.results['uploaded_file_name']}")

    # Step 2: Component Detection
    st.markdown("## Step 2: 🔍 Component Detection")
    detection_container = st.container()
    
    # Always show previous detection results if they exist
    if st.session_state.results['annotated_image'] is not None:
        with detection_container:
            st.image(st.session_state.results['annotated_image'], caption="Component Detection with Confidence Scores", use_container_width=True)
            
            # Display component statistics
            if st.session_state.results['component_stats'] is not None:
                cols = st.columns(len(st.session_state.results['component_stats']))
                for col, (component, stats) in zip(cols, st.session_state.results['component_stats'].items()):
                    with col:
                        avg_conf = stats['total_conf'] / stats['count']
                        st.metric(
                            label=component,
                            value=stats['count'],
                            delta=f"Avg Conf: {avg_conf:.2f}"
                        )
    
    if st.button("Detect Components"):
        print("Detect Components button clicked")
        if st.session_state.results['original_image'] is None:
            print("No original image found in session state")
            st.warning("Please upload an image first")
        else:
            print("Original image found in session state")
            with st.spinner("Detecting components..."):
                # Get raw bounding boxes
                raw_bboxes = analyzer.bboxes(st.session_state.results['original_image'])
                print(f"Found {len(raw_bboxes)} raw bounding boxes")
                
                # Apply Non-Maximum Suppression
                bboxes = non_max_suppression_by_confidence(raw_bboxes, iou_threshold=0.6)
                print(f"After NMS: {len(bboxes)} bounding boxes")
                st.session_state.results['bboxes'] = bboxes # Store filtered bboxes
                print("Stored bboxes in session state")
                
                # Summarize based on filtered bboxes
                detection_summary = summarize_components(bboxes)
                
                with detection_container:
                    # Display annotated image with confidence scores using filtered bboxes
                    annotated_image = st.session_state.results['original_image'].copy()
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
                    
                    st.session_state.results['annotated_image'] = annotated_image
                    st.image(annotated_image, caption="Component Detection with Confidence Scores", use_container_width=True)
                    
                    # Parse and display component counts
                    component_stats = {}
                    for bbox in bboxes:
                        name = bbox['class']
                        conf = bbox['confidence']
                        if name not in component_stats:
                            component_stats[name] = {'count': 0, 'total_conf': 0}
                        component_stats[name]['count'] += 1
                        component_stats[name]['total_conf'] += conf
                    
                    st.session_state.results['component_stats'] = component_stats
                    
                    cols = st.columns(len(component_stats))
                    for col, (component, stats) in zip(cols, component_stats.items()):
                        with col:
                            avg_conf = stats['total_conf'] / stats['count']
                            st.metric(
                                label=component,
                                value=stats['count'],
                                delta=f"Avg Conf: {avg_conf:.2f}"
                            )

    # Step 3: Node Analysis
    st.markdown("## Step 3: 🔗 Node Analysis")
    node_container = st.container()
    
    # Debug prints
    print("Current state of results:", {k: "Present" if v is not None else "None" for k, v in st.session_state.results.items()})
    
    # Always show previous node analysis results if they exist
    if st.session_state.results['node_visualization'] is not None:
        print("Showing previous node analysis results")
        with node_container:
            st.image(st.session_state.results['node_visualization'], caption="Node Connections", use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.results['node_mask'], caption="Node Mask")
            with col2:
                st.image(st.session_state.results['contour_image'], caption=f"Detected Nodes: {len(st.session_state.results['nodes'])}")
    
    if st.button("Analyze Nodes"):
        print("Analyze Nodes button clicked")
        with st.spinner("Analyzing nodes..."):
            if st.session_state.results['bboxes'] is None:
                print("No bboxes found in session state")
                st.warning("Please detect components first")
            else:
                print(f"Found {len(st.session_state.results['bboxes'])} bboxes in session state")
                try:
                    # Get emptied mask and show it for debugging
                    emptied_mask = analyzer.get_emptied_mask(st.session_state.results['original_image'], st.session_state.results['bboxes'])
                    st.image(emptied_mask, caption="Debug: Emptied Mask", use_container_width=True)
                    
                    # Get enhanced image and show it
                    enhanced = analyzer.enhance_lines(emptied_mask)
                    st.image(enhanced, caption="Debug: Enhanced Lines", use_container_width=True)
                    
                    nodes, emptied_mask, enhanced, contour_image, corners_image, final_visualization = analyzer.get_node_connections(
                        st.session_state.results['original_image'], 
                        st.session_state.results['bboxes']
                    )
                    
                    st.session_state.results['nodes'] = nodes
                    st.session_state.results['node_visualization'] = final_visualization
                    st.session_state.results['node_mask'] = emptied_mask
                    st.session_state.results['contour_image'] = contour_image
                    print(f"Node analysis complete. Found {len(nodes)} nodes")
                    
                    with node_container:
                        st.image(final_visualization, caption="Node Connections", use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(emptied_mask, caption="Node Mask")
                        with col2:
                            st.image(contour_image, caption=f"Detected Nodes: {len(nodes)}")
                except Exception as e:
                    print(f"Error during node analysis: {str(e)}")
                    st.error(f"Error during node analysis: {str(e)}")
                    import traceback
                    print("Full traceback:")
                    print(traceback.format_exc())

    # Step 4: Netlist Generation
    st.markdown("## Step 4: 📝 Netlist Generation")
    netlist_container = st.container()
    
    # Always show previous netlist results if they exist
    if st.session_state.results['netlist_text'] is not None:
        with netlist_container:
            st.markdown("### Initial Netlist")
            valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.results['nodes'])
            valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
            st.code(valueless_netlist_text, language="python")
            
            st.markdown("### Final Netlist with Component Values")
            st.code(st.session_state.results['netlist_text'], language="python")
    
    if st.button("Generate Netlist"):
        with st.spinner("Generating netlist..."):
            if st.session_state.results['nodes'] is not None:
                with netlist_container:
                    # Initial Netlist
                    st.markdown("### Initial Netlist")
                    valueless_netlist = analyzer.generate_netlist_from_nodes(st.session_state.results['nodes'])
                    valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
                    st.code(valueless_netlist_text, language="python")
                    
                    # Final Netlist
                    st.markdown("### Final Netlist with Component Values")
                    netlist = deepcopy(valueless_netlist)
                    enum_img, bbox_ids = analyzer.enumerate_components(st.session_state.results['original_image'], st.session_state.results['bboxes'])
                    
                    gemini_info = gemini_labels(enum_img)
                    analyzer.fix_netlist(netlist, gemini_info)
                    netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])
                    st.code(netlist_text, language="python")
                    
                    st.session_state.results['netlist'] = netlist
                    st.session_state.results['netlist_text'] = netlist_text
            else:
                st.warning("Please analyze nodes first")

    # Step 5: SPICE Analysis
    st.markdown("## Step 5: ⚡ SPICE Analysis")
    spice_container = st.container()
    if st.button("Run SPICE Analysis"):
        if st.session_state.results['netlist_text'] is not None:
            try:
                with spice_container:
                    net_text = '.title detected_circuit\n' + st.session_state.results['netlist_text']
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
                st.error(f"❌ SPICE Analysis Error: {str(e)}")
                st.info("💡 Tip: Check if all component values are properly detected and the circuit is properly connected.")
        else:
            st.warning("Please generate netlist first") 