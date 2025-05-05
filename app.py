import streamlit as st
import numpy as np
import cv2
import shutil
from src.utilities import summarize_components, gemini_labels
from src.circuit_analyzer import CircuitAnalyzer
from copy import deepcopy
from PySpice.Spice.Parser import SpiceParser
from pathlib import Path

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
MODEL_PATH = BASE_DIR / 'models/best_large_model_yolo.pt'

# Create necessary directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the circuit analyzer with error handling
@st.cache_resource
def load_circuit_analyzer():
    try:
        analyzer = CircuitAnalyzer(yolo_path=str(MODEL_PATH), debug=False)
        return analyzer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the circuit analyzer
analyzer = load_circuit_analyzer()

# Custom CSS with dark theme support
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            background-color: var(--background-color);
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
            border-radius: 0.5rem;
            background: linear-gradient(90deg, #1f77b4 0%, #2998ff 100%);
            border: none;
            color: white;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
        }
        .section-container {
            padding: 2rem;
            margin: 2rem 0;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1 {
            background: linear-gradient(90deg, #1f77b4 0%, #2998ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        h2, h3, h4 {
            color: var(--text-color);
            margin: 1.5rem 0;
            font-weight: 600;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #1f77b4 0%, #2998ff 100%);
        }
        img {
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚡ Circuit Analyzer")
    st.markdown("---")
    st.markdown("""
    ### Step-by-Step Guide
    1. 📤 Upload your circuit diagram
    2. 🔍 Review detected components
    3. 🔗 Check node connections
    4. 📝 Verify the netlist
    5. ⚡ View SPICE analysis
    """)
    st.markdown("---")
    st.info("💡 Supported formats: PNG, JPG, JPEG")

# Main content
st.title("Circuit Diagram Analysis Tool")
st.markdown("Upload your circuit diagram to get detailed component analysis and SPICE simulation results.")

# File upload section
uploaded_file = st.file_uploader(
    "Drag and drop your circuit diagram here",
    type=['png', 'jpg', 'jpeg'],
    help="For best results, use a clear image with good contrast"
)

if uploaded_file is not None:
    with st.spinner("Processing image..."):
        # Convert and display original image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Image Analysis Section
        st.markdown("## 📸 Image Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.markdown("### Image Details")
            st.markdown(f"- **Size**: {image.shape[1]}x{image.shape[0]} pixels")
            st.markdown(f"- **Format**: {uploaded_file.type}")
            st.markdown(f"- **Name**: {uploaded_file.name}")
        
        # Clear and save files
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / f'1.{uploaded_file.name.split(".")[-1]}'
        cv2.imwrite(str(image_path), image)

        # Component Detection Section
        st.markdown("## 🔍 Component Detection")
        progress_bar = st.progress(0)
        
        # Detection process
        bboxes = analyzer.bboxes(image)
        progress_bar.progress(33)
        detection_summary = summarize_components(bboxes)
        
        # Display annotated image
        annotated_image = analyzer.get_annotated(image, bboxes)
        st.image(annotated_image, use_container_width=True)
        
        # Parse and display component counts horizontally
        component_counts = {}
        for line in detection_summary.split('\n'):
            if line.startswith('Detected:'):
                components = line.replace('Detected:', '').strip().split(',')
                for comp in components:
                    if comp.strip():
                        count, *name_parts = comp.strip().split(' ')
                        component_name = ' '.join(name_parts)
                        component_counts[component_name] = int(count)
        
        # Create columns for each component metric
        cols = st.columns(len(component_counts))
        for col, (component, count) in zip(cols, component_counts.items()):
            with col:
                st.metric(
                    label=component,
                    value=count,
                    delta=None,
                )
        
        progress_bar.progress(66)
        
        # Node Analysis Section
        st.markdown("## 🔗 Node Analysis")
        nodes, emptied_mask, enhanced, contour_image, corners_image, final_visualization = analyzer.get_node_connections(image, bboxes)
        
        st.image(final_visualization, caption="Node Connections", use_container_width=True)
        
        # Debug visualization
        col1, col2 = st.columns(2)
        with col1:
            st.image(emptied_mask, caption="Node Mask")
        with col2:
            st.image(contour_image, caption=f"Detected Nodes: {len(nodes)}")
        
        progress_bar.progress(100)

        # Netlist Generation Section
        st.markdown("## 📝 Netlist Generation")
        
        # Initial Netlist
        st.markdown("### Initial Netlist")
        valueless_netlist = analyzer.generate_netlist_from_nodes(nodes)
        valueless_netlist_text = '\n'.join([analyzer.stringify_line(line) for line in valueless_netlist])
        st.code(valueless_netlist_text, language="python")
        
        # Final Netlist
        st.markdown("### Final Netlist with Component Values")
        netlist = deepcopy(valueless_netlist)
        enum_img, bbox_ids = analyzer.enumerate_components(image, bboxes)
        
        with st.spinner("Analyzing component values..."):
            gemini_info = gemini_labels(enum_img)
            analyzer.fix_netlist(netlist, gemini_info)
            netlist_text = '\n'.join([analyzer.stringify_line(line) for line in netlist])
        st.code(netlist_text, language="python")

        # SPICE Analysis Section
        st.markdown("## ⚡ SPICE Analysis")
        try:
            net_text = '.title detected_circuit\n' + netlist_text
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
            
            # Download Section
            st.markdown("## 📥 Download Results")
            st.markdown("""
            <style>
                div[data-testid="column"] {
                    display: flex;
                    justify-content: center;
                }
                
                div[data-testid="stDownloadButton"] {
                    width: 100%;
                    text-align: center;
                }
                
                div[data-testid="stDownloadButton"] button {
                    width: 100%;
                    padding: 0.5rem;
                }
            </style>
            """, unsafe_allow_html=True)
            
            download_cols = st.columns([1, 1, 1])
            with download_cols[0]:
                st.download_button(
                    label="📄 Download Netlist",
                    data=netlist_text,
                    file_name="circuit_netlist.txt",
                    mime="text/plain"
                )
            with download_cols[1]:
                st.download_button(
                    label="📊 Download Analysis",
                    data=str(analysis.nodes) + "\n\n" + str(analysis.branches),
                    file_name="spice_analysis.txt",
                    mime="text/plain"
                )
            with download_cols[2]:
                st.download_button(
                    label="🖼️ Annotated Circuit",
                    data=cv2.imencode('.png', annotated_image)[1].tobytes(),
                    file_name="annotated_circuit.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"❌ SPICE Analysis Error: {str(e)}")
            st.info("💡 Tip: Check if all component values are properly detected and the circuit is properly connected.") 