import os
import logging
import torch
from PIL import Image
import re
import time
import gradio as gr
import io
import numpy as np
from tqdm import tqdm

from src.utils import (
            create_annotated_image,
            calculate_component_stats,
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
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("circuit_analyzer_gradio")
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
logging.getLogger("gradio.routes").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# Set up base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR_GRADIO = os.path.join(BASE_DIR, 'static/uploads_gradio')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models/YOLO/best_large_model_yolo.pt')
SAM2_CONFIG_PATH_OBJ = os.path.join(BASE_DIR, 'models/configs/sam2.1_hiera_l.yaml')
SAM2_BASE_CHECKPOINT_PATH_OBJ = os.path.join(BASE_DIR, 'models/SAM2/sam2.1_hiera_large.pt')
SAM2_FINETUNED_CHECKPOINT_PATH_OBJ = os.path.join(BASE_DIR, 'models/SAM2/best_miou_model_SAM_latest.pth')
SAM2_CONFIG_PATH = "/" + os.path.abspath(SAM2_CONFIG_PATH_OBJ)
SAM2_BASE_CHECKPOINT_PATH = os.path.abspath(SAM2_BASE_CHECKPOINT_PATH_OBJ)
SAM2_FINETUNED_CHECKPOINT_PATH = os.path.abspath(SAM2_FINETUNED_CHECKPOINT_PATH_OBJ)

if not os.path.exists(UPLOAD_DIR_GRADIO):
    os.makedirs(UPLOAD_DIR_GRADIO, exist_ok=True)
sam2_dir = os.path.join(BASE_DIR, 'models/SAM2')
if not os.path.exists(sam2_dir):
    os.makedirs(sam2_dir, exist_ok=True)
logger.info(f"Created necessary directories: {UPLOAD_DIR_GRADIO}")

use_sam2_feature = True
missing_files_sam = []
if not os.path.exists(SAM2_CONFIG_PATH_OBJ): missing_files_sam.append(f"Config: {SAM2_CONFIG_PATH}")
if not os.path.exists(SAM2_BASE_CHECKPOINT_PATH_OBJ): missing_files_sam.append(f"Base Checkpoint: {SAM2_BASE_CHECKPOINT_PATH}")
if not os.path.exists(SAM2_FINETUNED_CHECKPOINT_PATH_OBJ): missing_files_sam.append(f"Fine-tuned Checkpoint: {SAM2_FINETUNED_CHECKPOINT_PATH}")

if missing_files_sam:
    logger.warning("One or more SAM2 model files not found:\n" + "\n".join(missing_files_sam) + "\nSAM2 features will be disabled.")
    use_sam2_feature = False
else:
    logger.info("All SAM2 model files found and will be used for SAM2 features.")

def load_circuit_analyzer():
    try:
        analyzer_obj = CircuitAnalyzer(
            yolo_path=str(YOLO_MODEL_PATH),
            sam2_config_path=str(SAM2_CONFIG_PATH),
            sam2_base_checkpoint_path=str(SAM2_BASE_CHECKPOINT_PATH),
            sam2_finetuned_checkpoint_path=str(SAM2_FINETUNED_CHECKPOINT_PATH),
            use_sam2=use_sam2_feature,
            debug=True
        )
        logger.info("CircuitAnalyzer loaded successfully.")
        return analyzer_obj
    except Exception as e:
        logger.error(f"Error loading CircuitAnalyzer model: {str(e)}", exc_info=True)
        return None

analyzer = load_circuit_analyzer()

# --- Helper Functions ---
def str_or_default(val, default="N/A"):
    return str(val) if val is not None else default

def format_detailed_timings(timings_dict):
    if not timings_dict:
        return []
    return [[step, f"{duration:.2f}"] for step, duration in timings_dict.items()]

def format_component_stats(stats_dict):
    if not stats_dict:
        return []
    data = []
    for component, stats in stats_dict.items():
        avg_conf = stats.get('total_conf', 0) / stats.get('count', 1) if stats.get('count', 0) > 0 else 0
        data.append([component, stats.get('count', 0), f"{avg_conf:.2f}"])
    return data

def format_image_info_md(active_results):
    if not active_results or 'original_image' not in active_results:
        return "*N/A*"
    name = active_results.get('uploaded_file_name', 'N/A')
    file_type = active_results.get('uploaded_file_type', 'N/A')
    img_array = active_results.get('original_image')
    h, w = (img_array.shape[0], img_array.shape[1]) if img_array is not None else ('N/A', 'N/A')
    pil_img_for_exif = active_results.get('original_image_pil')
    exif_str_list = []
    if pil_img_for_exif:
        try:
            if hasattr(pil_img_for_exif, 'info') and pil_img_for_exif.info.get('exif'):
                exif_str_list.append("- **EXIF Data**: Present (details not shown here)")
            else:
                exif_str_list.append("- **EXIF Data**: Not found or not parsed")
        except Exception as e:
            logger.warning(f"Could not extract basic EXIF info for Gradio display: {e}")
            exif_str_list.append("- **EXIF Data**: Error reading")
    exif_md = "\n".join(exif_str_list)
    return (f"- **Name**: {name}\n- **Type**: {file_type}\n- **Dimensions**: {h}x{w} pixels\n{exif_md}")

def _create_ui_updates(current_active_results, clear_all=False):
    updates = {}

    # Define default values for clearing
    default_none_updates = {
        original_image_display: None, annotated_image_display: None, initial_yolo_display: None,
        llama_debug_json: None, contour_image_display: None, sam2_image_display: None,
        emptied_mask_display: None, enhanced_mask_display: None, connection_points_display: None,
        node_viz_display: None, vlm_enum_image_display: None, vlm_raw_output_code: None,
        spice_node_voltages_json: None, spice_branch_currents_json: None, spice_phasor_plot: None
    }
    default_empty_list_updates = { component_stats_df: [], detailed_timings_df: [] }
    default_empty_string_updates = {
        initial_netlist_tb: "", final_netlist_tb: "", spice_sent_netlist_tb: ""
    }
    default_na_markdown_updates = {
        image_info_md: "*N/A*", cropping_details_md: "*N/A*", overall_time_md: "*N/A*"
    }
    status_markdown_defaults = {
        analysis_status_md: "*Status: Cleared. Waiting for new image.*",
        spice_status_md: "*Status: Waiting for analysis and SPICE run...*"
    }

    if clear_all:
        for component, value in default_none_updates.items(): updates[component] = gr.update(value=value)
        for component, value in default_empty_list_updates.items(): updates[component] = gr.update(value=value)
        for component, value in default_empty_string_updates.items(): updates[component] = gr.update(value=value)
        for component, value in default_na_markdown_updates.items(): updates[component] = gr.update(value=value)
        for component, value in status_markdown_defaults.items(): updates[component] = gr.update(value=value)
        return updates

    # Populate updates from current_active_results for Analysis Tab
    updates[original_image_display] = gr.update(value=current_active_results.get('original_image'))
    updates[annotated_image_display] = gr.update(value=current_active_results.get('annotated_image'))
    updates[image_info_md] = gr.update(value=format_image_info_md(current_active_results))
    crop_info = current_active_results.get('crop_debug_info', {})
    crop_applied_str = "Yes" if crop_info.get('crop_applied') else f"No ({crop_info.get('reason_for_no_crop', 'Unknown')})"
    crop_details_text = (f"- **Crop Applied**: {crop_applied_str}\n"
                         f"- **Original Dims**: {crop_info.get('original_image_dims', 'N/A')}\n"
                         f"- **Cropped Dims**: {crop_info.get('cropped_image_dims', 'N/A')}\n"
                         f"- **Defining BBox for Crop**: {crop_info.get('defining_bbox', 'N/A')}")
    updates[cropping_details_md] = gr.update(value=crop_details_text if crop_info else "*N/A*")
    updates[initial_yolo_display] = gr.update(value=current_active_results.get('initial_yolo_annotated_image')) # Changed key
    enriched_bboxes = [b for b in current_active_results.get('bboxes_after_reclass_before_llama', []) if 'semantic_direction' in b]
    updates[llama_debug_json] = gr.update(value=enriched_bboxes if enriched_bboxes else None)
    updates[component_stats_df] = gr.update(value=format_component_stats(current_active_results.get('component_stats')))
    updates[contour_image_display] = gr.update(value=current_active_results.get('contour_image'))
    updates[sam2_image_display] = gr.update(value=current_active_results.get('sam2_output'))
    updates[emptied_mask_display] = gr.update(value=current_active_results.get('node_mask'))
    updates[enhanced_mask_display] = gr.update(value=current_active_results.get('enhanced_mask'))
    updates[connection_points_display] = gr.update(value=current_active_results.get('connection_points_image'))
    updates[node_viz_display] = gr.update(value=current_active_results.get('node_visualization'))
    updates[initial_netlist_tb] = gr.update(value=current_active_results.get('valueless_netlist_text'))
    updates[final_netlist_tb] = gr.update(value=current_active_results.get('netlist_text', current_active_results.get('valueless_netlist_text')))
    updates[vlm_enum_image_display] = gr.update(value=current_active_results.get('enum_img'))
    updates[vlm_raw_output_code] = gr.update(value=current_active_results.get('vlm_stage2_output') or None)
    elapsed_time = current_active_results.get('elapsed_time')
    updates[overall_time_md] = gr.update(value=f"{elapsed_time:.2f} seconds" if elapsed_time is not None else "*N/A*")
    updates[detailed_timings_df] = gr.update(value=format_detailed_timings(current_active_results.get('detailed_timings')))

    # Populate SPICE Tab updates
    spice_dc_output = current_active_results.get('spice_dc_output', {})
    spice_ac_output = current_active_results.get('spice_ac_output', {})

    if spice_ac_output.get('status') == 'success':
        updates[spice_sent_netlist_tb] = gr.update(value=spice_ac_output.get('ac_netlist'))
        updates[spice_node_voltages_json] = gr.update(value=spice_ac_output.get('node_voltages'))
        updates[spice_branch_currents_json] = gr.update(value=spice_ac_output.get('branch_currents'))
        updates[spice_phasor_plot] = gr.update(value=spice_ac_output.get('phasor_plot_fig'))
    elif spice_dc_output.get('status') == 'success':
        updates[spice_sent_netlist_tb] = gr.update(value=spice_dc_output.get('dc_netlist'))
        updates[spice_node_voltages_json] = gr.update(value=spice_dc_output.get('node_voltages'))
        updates[spice_branch_currents_json] = gr.update(value=spice_dc_output.get('branch_currents'))
        updates[spice_phasor_plot] = gr.update(value=None) # Explicitly clear plot for DC
    else: # Clear if neither was successful or if error
        # Show netlist sent to simulator even on error, if available from the results dict
        updates[spice_sent_netlist_tb] = gr.update(value=spice_dc_output.get('dc_netlist',None) or spice_ac_output.get('ac_netlist',None) or "")
        updates[spice_node_voltages_json] = gr.update(value=None)
        updates[spice_branch_currents_json] = gr.update(value=None)
        updates[spice_phasor_plot] = gr.update(value=None)

    return updates

# --- Main Gradio UI Structure ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.cyan)) as demo:
    active_results_state = gr.State({})
    gr.Markdown("<h1>⚡ CircuitVision - Gradio Edition ⚡</h1>")
    gr.Markdown("AI-powered circuit diagram analysis and SPICE simulation. Upload an image to start.")

    if analyzer:
        gr.Markdown("✅ **Circuit Analyzer models loaded successfully.**")
        if hasattr(analyzer, 'use_sam2') and analyzer.use_sam2: gr.Markdown("✅ SAM2 features are **enabled**.")
        else: gr.Markdown("⚠️ SAM2 features are **disabled**.")
    else:
        gr.Markdown("🚨 **CRITICAL: Circuit Analyzer models FAILED to load.**")

    with gr.Tabs() as main_tabs:
        with gr.TabItem("🔬 Analysis", id="analysis_tab"):
            with gr.Row():
                with gr.Column(scale=1): image_upload_input = gr.Image(type="pil", label="Upload Circuit Diagram", elem_id="image_upload", sources=["upload", "clipboard"])
                with gr.Column(scale=1, min_width=200):
                    analyze_circuit_btn = gr.Button("Analyze Circuit", variant="primary")
                    analysis_status_md = gr.Markdown("*Status: Waiting for image and analysis start...*")
            with gr.Accordion("⏱️ Analysis Timings", open=False) as analysis_timings_accordion:
                overall_time_md = gr.Markdown(label="Overall Analysis Time", value="*N/A*")
                detailed_timings_df = gr.DataFrame(label="Detailed Step Timings", headers=["Step", "Duration (s)"], col_count=(2, "fixed"), interactive=False, value=None)
            with gr.Row():
                original_image_display = gr.Image(label="Original Image (Uploaded/Processed)", interactive=False)
                annotated_image_display = gr.Image(label="Component Detections / Annotated Image", interactive=False)
            with gr.Accordion("🖼️ Image & Detection Details", open=False) as img_details_accordion:
                image_info_md = gr.Markdown(label="Original Image Info", value="*N/A*")
                cropping_details_md = gr.Markdown(label="Cropping Details", value="*N/A*")
                initial_yolo_display = gr.Image(label="Initial YOLO Detections (Original Image)", interactive=False)
                with gr.Accordion("LLaMA Direction Debug (Raw Enriched BBoxes)", open=False) as llama_debug_accordion:
                    llama_debug_json = gr.JSON(label="LLaMA Enriched BBoxes Data")
            component_stats_df = gr.DataFrame(label="Component Statistics", headers=["Component", "Count", "Avg. Confidence"], col_count=(3, "fixed"), interactive=False, value=None)
            with gr.Accordion("🔗 Node Analysis Debug Images", open=False) as node_debug_accordion:
                with gr.Row():
                    contour_image_display = gr.Image(label="Contours", interactive=False)
                    sam2_image_display = gr.Image(label="SAM2 Segmentation", interactive=False)
                with gr.Row():
                    emptied_mask_display = gr.Image(label="Emptied Mask", interactive=False)
                    enhanced_mask_display = gr.Image(label="Enhanced Mask", interactive=False)
                connection_points_display = gr.Image(label="Connection Points", interactive=False)
                node_viz_display = gr.Image(label="Final Node Connections", interactive=False)
            with gr.Row():
                initial_netlist_tb = gr.Textbox(label="Initial Netlist (Valueless)", lines=10, interactive=False)
                final_netlist_tb = gr.Textbox(label="Final Netlist (Editable for SPICE)", lines=10, interactive=True, info="This netlist is used for SPICE simulation.")
            generate_final_netlist_btn = gr.Button("Generate Final Netlist (with VLM)")
            with gr.Accordion("🤖 VLM Debugging (Final Netlist Stage)", open=False) as vlm_debug_accordion:
                vlm_enum_image_display = gr.Image(label="Image Sent to VLM (Enumerated)", interactive=False)
                vlm_raw_output_code = gr.Code(label="VLM Raw Output (JSON)", language="json", interactive=False)

        with gr.TabItem("⚡ SPICE Simulation", id="spice_tab"):
            gr.Markdown("SPICE analysis uses the **'Final Netlist (Editable for SPICE)'** from the '🔬 Analysis' tab.")
            with gr.Row():
                with gr.Column(scale=1):
                    spice_analysis_type_radio = gr.Radio(label="SPICE Analysis Type", choices=["DC (.op)", "AC (.ac)"], value="DC (.op)", elem_id="spice_type_radio")
                    spice_ac_freq_num = gr.Number(label="AC Frequency (Hz)", value=60.0, interactive=True, visible=False, info="Specify frequency for AC analysis.")
                with gr.Column(scale=1, min_width=200):
                     run_spice_btn = gr.Button("Run SPICE Analysis", variant="primary")
                     spice_status_md = gr.Markdown("*Status: Waiting for SPICE run...*")
            with gr.Accordion("📈 SPICE Debug and Results", open=True) as spice_debug_accordion:
                spice_sent_netlist_tb = gr.Textbox(label="Actual SPICE Netlist Sent to Simulator", lines=7, interactive=False)
                with gr.Row():
                    spice_node_voltages_json = gr.JSON(label="Node Voltages")
                    spice_branch_currents_json = gr.JSON(label="Branch Currents")
                spice_phasor_plot = gr.Plot(label="AC Phasor Plot")

    analysis_outputs_ordered_list = [
        active_results_state, original_image_display, annotated_image_display, image_info_md,
        cropping_details_md, initial_yolo_display, llama_debug_json, component_stats_df,
        contour_image_display, sam2_image_display, emptied_mask_display, enhanced_mask_display,
        connection_points_display, node_viz_display, initial_netlist_tb, final_netlist_tb,
        vlm_enum_image_display, vlm_raw_output_code, overall_time_md, detailed_timings_df,
        analysis_status_md, spice_sent_netlist_tb, spice_node_voltages_json,
        spice_branch_currents_json, spice_phasor_plot, spice_status_md
    ]

    def handle_image_upload(uploaded_file_obj, current_active_results_state_dict, progress=gr.Progress(track_tqdm=True)):
        if uploaded_file_obj is None:
            new_active_results = {}
            updates = _create_ui_updates({}, clear_all=True)
            return [new_active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]
        logger.info("Handling image upload...")
        progress(0, desc="Processing uploaded image...")
        img_byte_arr = io.BytesIO(); uploaded_file_obj.save(img_byte_arr, format='PNG')
        uploaded_file_data = img_byte_arr.getvalue()
        file_name_attr = getattr(uploaded_file_obj, 'name', None)
        file_name = os.path.basename(file_name_attr) if file_name_attr else "uploaded_image.png"
        file_type = "image/png"
        try:
            new_active_results = process_new_upload(uploaded_file_data, file_name, file_type, UPLOAD_DIR_GRADIO, logger)
            progress(1, desc="Image processed.")
            updates = _create_ui_updates(new_active_results) # Initial population after upload
            updates[analysis_status_md] = gr.update(value=f"*Status: Image '{file_name}' loaded. Ready for analysis.*")
            # image_info_md is already handled by _create_ui_updates
        except Exception as e:
            logger.error(f"Error processing uploaded image: {e}", exc_info=True)
            new_active_results = {}; updates = _create_ui_updates({}, clear_all=True)
            updates[analysis_status_md] = gr.update(value=f"*Error processing image: {e}*")
        return [new_active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

    def run_full_analysis_callback(current_active_results_state_dict, progress=gr.Progress(track_tqdm=True)):
        if not isinstance(current_active_results_state_dict, dict):
            error_updates = _create_ui_updates({}, clear_all=True)
            error_updates[analysis_status_md] = gr.update(value="*Error: Internal state invalid.*")
            return [current_active_results_state_dict] + [error_updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]
        active_results = current_active_results_state_dict.copy()
        if not active_results.get('original_image_pil') or not active_results.get('image_path'):
            updates = _create_ui_updates(active_results); updates[analysis_status_md] = gr.update(value="*Error: Image data not loaded.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]
        if analyzer is None:
            updates = _create_ui_updates(active_results); updates[analysis_status_md] = gr.update(value="*Error: Analyzer not loaded.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

        logger.info("Starting full circuit analysis...")
        active_results['detailed_timings'] = {}; overall_start_time = time.time()
        # Clear previous SPICE results from state when starting a new full analysis
        active_results.pop('spice_dc_output', None)
        active_results.pop('spice_ac_output', None)

        try:
            steps = [
                ("Initial Detection & Enrichment", lambda: run_initial_detection_and_enrichment(analyzer, active_results, active_results['detailed_timings'], logger)),
                ("Segmentation & Cropping", lambda: run_segmentation_and_cropping(analyzer, active_results, active_results['detailed_timings'], logger)),
                ("Annotated Image & Stats", lambda: (
                    setattr(active_results, 'annotated_image', create_annotated_image(active_results['image_for_analysis'], active_results['bboxes'])),
                    setattr(active_results, 'component_stats', calculate_component_stats(active_results['bboxes'])),
                    # For initial_yolo_display
                    setattr(active_results, 'initial_yolo_annotated_image', create_annotated_image(active_results['original_image'], active_results.get('bboxes_orig_coords_nms', [])) if active_results.get('original_image') and active_results.get('bboxes_orig_coords_nms') else None)
                )),
                ("Node Analysis", lambda: run_node_analysis(analyzer, active_results['image_for_analysis'], active_results.get('cropped_sam_mask_for_nodes'), active_results['bboxes'], active_results, active_results['detailed_timings'], logger)),
                ("Initial Netlist Generation", lambda: run_initial_netlist_generation(analyzer, active_results.get('nodes'), active_results['image_for_analysis'], active_results['bboxes'], active_results, active_results['detailed_timings'], logger))
            ]
            for i, (step_name, step_func) in enumerate(tqdm(steps, desc="Analysis Pipeline")):
                progress((i + 1) / len(steps), desc=step_name); logger.info(f"Running step: {step_name}"); step_func()
            active_results['elapsed_time'] = time.time() - overall_start_time
            log_analysis_summary(active_results, logger, str(logger.level))
            logger.info("Full analysis completed.")
            updates = _create_ui_updates(active_results) # This will also clear SPICE fields correctly now
            updates[analysis_status_md] = gr.update(value=f"*Status: Analysis complete in {active_results['elapsed_time']:.2f}s.*")
        except Exception as e:
            logger.error(f"Error during full analysis pipeline: {e}", exc_info=True)
            active_results['elapsed_time'] = time.time() - overall_start_time
            updates = _create_ui_updates(active_results)
            updates[analysis_status_md] = gr.update(value=f"*Error during analysis: {e}*")
            updates[analysis_timings_accordion] = gr.update(open=True)
        return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

    def generate_final_netlist_callback(current_active_results_state_dict, progress=gr.Progress(track_tqdm=True)):
        if not isinstance(current_active_results_state_dict, dict):
            error_updates = _create_ui_updates({}, clear_all=True)
            error_updates[analysis_status_md] = gr.update(value="*Error: Internal state invalid for VLM.*")
            return [current_active_results_state_dict] + [error_updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

        active_results = current_active_results_state_dict.copy()
        if not all(k in active_results for k in ['enum_img', 'bbox_ids', 'netlist']):
            updates = _create_ui_updates(active_results)
            updates[analysis_status_md] = gr.update(value="*Error: Prerequisite data for VLM missing. Run full analysis.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]
        if analyzer is None:
            updates = _create_ui_updates(active_results); updates[analysis_status_md] = gr.update(value="*Error: Analyzer not loaded for VLM.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

        logger.info("Generating Final Netlist with VLM...")
        progress(0.5, desc="VLM processing...")
        success_flag = handle_final_netlist_generation(analyzer, active_results, logger)
        progress(1, desc="VLM processing complete.")

        updates = _create_ui_updates(active_results)
        if success_flag:
            updates[analysis_status_md] = gr.update(value="*Status: Final netlist generated successfully.*")
        else:
            updates[analysis_status_md] = gr.update(value="*Status: Final netlist generation with VLM issues (check logs).*")
        return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

    def run_spice_analysis_callback(current_active_results_state_dict, final_netlist_content, analysis_type, ac_frequency_str, progress=gr.Progress(track_tqdm=True)):
        active_results = current_active_results_state_dict.copy() if isinstance(current_active_results_state_dict, dict) else {}

        if not final_netlist_content:
            updates = _create_ui_updates(active_results)
            updates[spice_status_md] = gr.update(value="*Error: Final Netlist is empty for SPICE.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]
        if analyzer is None:
            updates = _create_ui_updates(active_results); updates[spice_status_md] = gr.update(value="*Error: Analyzer not loaded for SPICE.*")
            return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

        logger.info(f"Running SPICE Analysis: Type={analysis_type}, Freq={ac_frequency_str if analysis_type == 'AC (.ac)' else 'N/A'}")
        progress(0.5, desc=f"Executing {analysis_type} SPICE simulation...")

        spice_results = None
        if analysis_type == "DC (.op)":
            spice_results = perform_dc_spice_analysis(final_netlist_content, logger)
            active_results['spice_dc_output'] = spice_results
            active_results.pop('spice_ac_output', None) # Clear previous AC results
        elif analysis_type == "AC (.ac)":
            try:
                ac_frequency = float(ac_frequency_str)
                if ac_frequency <= 0: raise ValueError("AC frequency must be positive.")
                spice_results = perform_ac_spice_analysis(active_results, analyzer, ac_frequency, logger)
                active_results['spice_ac_output'] = spice_results
                active_results.pop('spice_dc_output', None) # Clear previous DC results
            except ValueError as ve:
                logger.error(f"Invalid AC frequency: {ac_frequency_str}. Error: {ve}")
                spice_results = {"status": "error", "message": f"Invalid AC frequency: {ve}"}
                active_results['spice_ac_output'] = spice_results

        progress(1, desc="SPICE simulation complete.")
        updates = _create_ui_updates(active_results)

        if spice_results and spice_results.get('status') == 'success':
            updates[spice_status_md] = gr.update(value=f"*Status: {analysis_type} SPICE analysis successful.*")
        else:
            err_msg = spice_results.get('message', 'Unknown SPICE error') if spice_results else 'Simulation failed to run'
            updates[spice_status_md] = gr.update(value=f"*Status: {analysis_type} SPICE analysis failed: {err_msg}*")
            updates[spice_debug_accordion] = gr.update(open=True)

        return [active_results] + [updates.get(comp, gr.update()) for comp in analysis_outputs_ordered_list[1:]]

    def toggle_ac_frequency_visibility(analysis_type_choice):
        return gr.update(visible=(analysis_type_choice == "AC (.ac)"))

    # --- Connect Callbacks ---
    image_upload_input.upload(fn=handle_image_upload, inputs=[image_upload_input, active_results_state], outputs=analysis_outputs_ordered_list, show_progress="full")
    analyze_circuit_btn.click(fn=run_full_analysis_callback, inputs=[active_results_state], outputs=analysis_outputs_ordered_list, show_progress="full")
    generate_final_netlist_btn.click(fn=generate_final_netlist_callback, inputs=[active_results_state], outputs=analysis_outputs_ordered_list, show_progress="full")
    run_spice_btn.click(fn=run_spice_analysis_callback, inputs=[active_results_state, final_netlist_tb, spice_analysis_type_radio, spice_ac_freq_num], outputs=analysis_outputs_ordered_list, show_progress="full")
    spice_analysis_type_radio.change(fn=toggle_ac_frequency_visibility, inputs=[spice_analysis_type_radio], outputs=[spice_ac_freq_num])


if __name__ == "__main__":
    if analyzer: logger.info("Circuit Analyzer loaded. Launching Gradio app.")
    else: logger.error("Circuit Analyzer failed to load. Gradio app will launch, but may not function correctly.")
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
    logger.info("Gradio app launched.")
