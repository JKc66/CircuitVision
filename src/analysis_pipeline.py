import os
import shutil
import time
import logging
from copy import deepcopy
import numpy as np
import cv2
# import streamlit as st # Removed Streamlit import
from PIL import Image, ImageOps
from src.utils import non_max_suppression_by_confidence, gemini_labels_openrouter, non_max_suppression_by_area
import io # Added for io.BytesIO

def process_new_upload(uploaded_file_data, uploaded_file_name, uploaded_file_type, upload_dir_path, app_logger):
    """Handles the processing of a newly uploaded image file."""
    app_logger.info(f"NEW IMAGE UPLOADED: {uploaded_file_name}")
    app_logger.info("="*80)

    active_results = {} # Initialize local active_results

    # Convert image data (bytes) to OpenCV format
    image_np_array = np.frombuffer(uploaded_file_data, np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    if image is None:
        app_logger.error(f"Could not decode image: {uploaded_file_name}")
        raise ValueError(f"Could not decode image: {uploaded_file_name}. Ensure it is a valid image format.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    disk_image_path = os.path.join(upload_dir_path, uploaded_file_name)

    active_results.update({
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': image,
        'uploaded_file_type': uploaded_file_type,
        'uploaded_file_name': uploaded_file_name,
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
    })

    # Clear and save files
    if os.path.exists(upload_dir_path):
        shutil.rmtree(upload_dir_path)
    os.makedirs(upload_dir_path, exist_ok=True)

    # Save using PIL to preserve EXIF data
    # Use uploaded_file_data (bytes) for PIL
    try:
        pil_image_for_exif = Image.open(io.BytesIO(uploaded_file_data))
    except Exception as e:
        app_logger.error(f"Could not open image data with PIL: {e}")
        # Fallback: use the cv2-opened image if PIL fails, though EXIF will be lost
        pil_image_for_exif = Image.fromarray(active_results['original_image'])


    active_results['original_image_pil'] = pil_image_for_exif

    app_logger.info(f"Checking EXIF orientation for {uploaded_file_name}")
    final_image_to_save = pil_image_for_exif
    try:
        exif = pil_image_for_exif._getexif()
        if exif and 0x0112 in exif:
            orientation = exif[0x0112]
            app_logger.info(f"Found orientation tag: {orientation}")
            if orientation != 1:
                app_logger.info(f"Auto-rotating image with orientation {orientation}")
                final_image_to_save = ImageOps.exif_transpose(pil_image_for_exif)
                active_results['original_image'] = np.array(final_image_to_save.convert("RGB")) # Ensure RGB for numpy array
    except Exception as e:
        app_logger.error(f"Error checking/rotating image based on EXIF: {str(e)}")

    exif_bytes_to_save = final_image_to_save.info.get('exif')
    image_format_for_save = final_image_to_save.format or uploaded_file_type.split('/')[-1].upper()
    # Ensure format is one PIL can save, default to PNG if unknown/problematic
    if not image_format_for_save or image_format_for_save.lower() not in ['jpeg', 'jpg', 'png', 'tiff', 'gif', 'bmp']:
        app_logger.warning(f"Unknown or unsaveable format '{image_format_for_save}', defaulting to PNG for saving.")
        image_format_for_save = 'PNG'
        # Update disk_image_path if extension changes (e.g. from .jfif to .png)
        base, _ = os.path.splitext(disk_image_path)
        disk_image_path = base + ".png"
        active_results['image_path'] = disk_image_path # Update stored path


    try:
        if exif_bytes_to_save:
            final_image_to_save.save(disk_image_path, format=image_format_for_save, exif=exif_bytes_to_save)
            app_logger.info(f"Saved image {disk_image_path} with EXIF data.")
        else:
            final_image_to_save.save(disk_image_path, format=image_format_for_save)
            app_logger.info(f"Saved image {disk_image_path} without EXIF data.")
    except Exception as e_save:
        app_logger.error(f"Error saving image to {disk_image_path} with format {image_format_for_save}: {e_save}")
        # Attempt to save as PNG as a fallback if the original format failed
        if image_format_for_save.upper() != 'PNG':
            app_logger.info("Attempting to save as PNG as fallback...")
            try:
                base, _ = os.path.splitext(disk_image_path)
                disk_image_path_png = base + ".png"
                active_results['image_path'] = disk_image_path_png # Update stored path
                # Convert to RGB before saving as PNG if it's RGBA or other mode not directly supported by some PNG writers
                final_image_to_save.convert("RGB").save(disk_image_path_png, format="PNG")
                app_logger.info(f"Successfully saved image as PNG: {disk_image_path_png}")
            except Exception as e_png_save:
                app_logger.error(f"Failed to save image as PNG fallback: {e_png_save}")
                raise IOError(f"Failed to save image {uploaded_file_name} in any format.") from e_png_save
        else:
            raise IOError(f"Failed to save image {uploaded_file_name} as {image_format_for_save}.") from e_save


    # Removed Streamlit-specific session state modifications
    # st.session_state.final_netlist_generated = False
    # st.session_state.start_analysis_triggered = True
    # st.session_state.analysis_in_progress = True
    return active_results

def run_initial_detection_and_enrichment(current_analyzer, active_results_dict, detailed_timings_dict, app_logger):
    """Performs initial component detection (YOLO) and semantic enrichment (LLaMA)."""
    step_start_time_yolo = time.time()
    raw_bboxes_orig_coords = []
    if active_results_dict.get('original_image') is not None:
        app_logger.debug("Step 1A: Detecting components with YOLO on original image...")
        raw_bboxes_orig_coords = current_analyzer.bboxes(active_results_dict['original_image'])
        app_logger.debug(f"Detected {len(raw_bboxes_orig_coords)} raw bounding boxes on original image")

        bboxes_orig_coords_nms = non_max_suppression_by_confidence(raw_bboxes_orig_coords, iou_threshold=0.6)
        app_logger.debug(f"After NMS: {len(bboxes_orig_coords_nms)} bounding boxes on original image")
        active_results_dict['bboxes_orig_coords_nms'] = bboxes_orig_coords_nms
    else:
        app_logger.error("No original image found for YOLO detection during analysis.")
        raise ValueError("Original image not available for YOLO analysis.")
    detailed_timings_dict['YOLO Component Detection'] = time.time() - step_start_time_yolo

    app_logger.info("Attempting preliminary reclassification of 'terminal' components...")
    if 'original_image_pil' in active_results_dict and active_results_dict['original_image_pil'] is not None:
        image_pil_for_reclass = active_results_dict['original_image_pil']
        # Ensure image is RGB for reclassification if it's not already
        image_rgb_for_reclass = np.array(image_pil_for_reclass.convert("RGB"))
        current_analyzer.reclassify_terminals_based_on_connectivity(image_rgb_for_reclass, active_results_dict['bboxes_orig_coords_nms'])
        app_logger.info("Preliminary reclassification of 'terminal' components completed.")
    else:
        app_logger.warning("Skipping terminal reclassification: 'original_image_pil' not found.")

    if active_results_dict.get('bboxes_orig_coords_nms'):
         active_results_dict['bboxes_after_reclass_before_llama'] = deepcopy(active_results_dict['bboxes_orig_coords_nms'])

    can_enrich = (
        active_results_dict.get('bboxes_orig_coords_nms') and
        active_results_dict.get('original_image') is not None and
        hasattr(current_analyzer, 'groq_client') and current_analyzer.groq_client
    )
    if can_enrich:
        app_logger.info("Attempting to enrich component bboxes with semantic directions from LLaMA...")
        step_start_time_llama_enrich = time.time()
        try:
            current_analyzer._enrich_bboxes_with_directions(
                active_results_dict['original_image'],
                active_results_dict['bboxes_orig_coords_nms']
            )
            app_logger.info("Semantic direction enrichment step completed.")
        except Exception as e_enrich:
            app_logger.error(f"Error during LLaMA semantic direction enrichment: {e_enrich}")
            app_logger.warning(f"Could not determine semantic directions for some components using LLaMA: {e_enrich}") # Replaced st.warning
        finally:
            detailed_timings_dict['LLaMA Direction Enrichment'] = time.time() - step_start_time_llama_enrich
    elif active_results_dict.get('bboxes_orig_coords_nms') and active_results_dict.get('original_image') is not None:
        app_logger.warning("Skipping LLaMA semantic direction enrichment: Groq client not available or other issue.")

    return active_results_dict.get('bboxes_orig_coords_nms')

def run_segmentation_and_cropping(current_analyzer, active_results_dict, detailed_timings_dict, app_logger):
    """Performs YOLO-based cropping first, then SAM2 segmentation on the (potentially) cropped image."""
    original_image = active_results_dict['original_image']
    bboxes_from_initial_detection = active_results_dict.get('bboxes_orig_coords_nms', [])

    step_start_time_yolo_crop = time.time()
    app_logger.debug("Attempting YOLO-based cropping...")
    image_after_yolo_crop, bboxes_after_yolo_crop, crop_debug_info = current_analyzer.crop_image_and_adjust_bboxes(
        original_image,
        deepcopy(bboxes_from_initial_detection),
        padding=80
    )
    active_results_dict['crop_debug_info'] = crop_debug_info
    detailed_timings_dict['YOLO-based Image Cropping'] = time.time() - step_start_time_yolo_crop

    if crop_debug_info and crop_debug_info.get('crop_applied'):
        app_logger.info(f"YOLO-based crop applied. New image dimensions: {image_after_yolo_crop.shape[:2]}")
    else:
        app_logger.info(f"YOLO-based crop NOT applied. Reason: {crop_debug_info.get('reason_for_no_crop', 'Unknown')}. Using original image dimensions.")

    active_results_dict['image_for_analysis'] = image_after_yolo_crop
    active_results_dict['bboxes'] = bboxes_after_yolo_crop

    step_start_time_sam = time.time()
    binary_sam_mask_on_yolo_cropped_img = None
    sam2_colored_display_output = None

    if current_analyzer.use_sam2:
        app_logger.debug("Step 2B: Performing SAM2 segmentation on the (potentially YOLO-cropped) image_for_analysis...")
        binary_sam_mask_on_yolo_cropped_img, sam2_colored_display_output, _ = current_analyzer.segment_with_sam2(
            image_after_yolo_crop
        )
        active_results_dict['sam2_output'] = sam2_colored_display_output
        if binary_sam_mask_on_yolo_cropped_img is not None:
            app_logger.info(f"SAM2 segmentation successful on YOLO-cropped image. Mask shape: {binary_sam_mask_on_yolo_cropped_img.shape}")
        else:
            app_logger.warning("SAM2 segmentation on YOLO-cropped image did not return a binary mask.")
    elif not current_analyzer.use_sam2:
        app_logger.warning("SAM2 is disabled. Skipping SAM2 segmentation.")

    detailed_timings_dict['SAM2 Segmentation on YOLO-Cropped Image'] = time.time() - step_start_time_sam
    active_results_dict['cropped_sam_mask_for_nodes'] = binary_sam_mask_on_yolo_cropped_img

    return image_after_yolo_crop, bboxes_after_yolo_crop, binary_sam_mask_on_yolo_cropped_img

def run_node_analysis(current_analyzer, img_for_analysis, crpd_sam_mask_nodes, bboxes_for_analysis_nodes, active_results_dict, detailed_timings_dict, app_logger):
    """Performs node connection analysis using SAM mask and bboxes."""
    step_start_time_nodes = time.time()
    nodes_identified = None
    if bboxes_for_analysis_nodes is not None and crpd_sam_mask_nodes is not None and current_analyzer.use_sam2:
        try:
            app_logger.debug("Step 2: Analyzing node connections using cropped SAM mask and adjusted bboxes...")
            nodes_identified, emptied_mask, enhanced, contour_image, final_visualization, conn_points_img = current_analyzer.get_node_connections(
                img_for_analysis,
                crpd_sam_mask_nodes,
                bboxes_for_analysis_nodes
            )
            app_logger.debug(f"Node analysis completed: {len(nodes_identified) if nodes_identified is not None else 0} nodes identified")
            active_results_dict['nodes'] = nodes_identified
            active_results_dict['node_visualization'] = final_visualization
            active_results_dict['node_mask'] = emptied_mask
            active_results_dict['enhanced_mask'] = enhanced
            active_results_dict['contour_image'] = contour_image
            active_results_dict['connection_points_image'] = conn_points_img
        except Exception as e:
            error_msg = f"Error during node analysis: {str(e)}"
            app_logger.error(error_msg) # Replaced st.error
            app_logger.warning("Continuing execution despite node analysis error")
    elif not current_analyzer.use_sam2:
        app_logger.warning("Node analysis skipped: SAM2 is disabled.") # Replaced st.warning
    else:
        app_logger.error("Node analysis skipped: Bounding boxes or cropped SAM2 mask not available.") # Replaced st.error

    detailed_timings_dict['Node Analysis'] = time.time() - step_start_time_nodes
    return nodes_identified

def run_initial_netlist_generation(current_analyzer, identified_nodes, img_for_analysis, bboxes_for_enum, active_results_dict, detailed_timings_dict, app_logger):
    """Generates the initial valueless netlist and enumerates components."""
    step_start_time_netlist = time.time()
    valueless_netlist_generated = None

    if identified_nodes is not None:
        try:
            app_logger.debug("Step 3: Generating initial netlist...")
            valueless_netlist_generated = current_analyzer.generate_netlist_from_nodes(identified_nodes)
            valueless_netlist_text = '\n'.join([current_analyzer.stringify_line(line) for line in valueless_netlist_generated])
            app_logger.debug(f"Generated initial netlist with {len(valueless_netlist_generated)} components")

            active_results_dict['netlist'] = valueless_netlist_generated
            active_results_dict['valueless_netlist_text'] = valueless_netlist_text
            active_results_dict['netlist_text'] = valueless_netlist_text # Initially, netlist_text is the valueless one
            # Removed: st.session_state.editable_netlist_content = valueless_netlist_text

            try:
                app_logger.debug("Generating initial netlist WITHOUT LLaMA directions for comparison...")
                nodes_copy_for_no_llama = deepcopy(identified_nodes)
                for node_data in nodes_copy_for_no_llama:
                    for component_in_node in node_data.get('components', []):
                        component_in_node['semantic_direction'] = "UNKNOWN"
                valueless_netlist_no_llama_dir = current_analyzer.generate_netlist_from_nodes(nodes_copy_for_no_llama)
                valueless_netlist_text_no_llama_dir = '\n'.join([current_analyzer.stringify_line(line) for line in valueless_netlist_no_llama_dir])
                active_results_dict['valueless_netlist_text_no_llama_dir'] = valueless_netlist_text_no_llama_dir
                app_logger.debug(f"Generated initial netlist WITHOUT LLaMA directions: {len(valueless_netlist_no_llama_dir)} components")
            except Exception as e_no_llama:
                app_logger.error(f"Error generating netlist without LLaMA directions: {e_no_llama}")
                active_results_dict['valueless_netlist_text_no_llama_dir'] = "Error generating this version."

            app_logger.debug("Enumerating components for later Gemini labeling...")
            enum_img, bbox_ids_with_visual_enum = current_analyzer.enumerate_components(
                img_for_analysis,
                deepcopy(bboxes_for_enum)
            )
            active_results_dict['enum_img'] = enum_img
            active_results_dict['bbox_ids'] = bbox_ids_with_visual_enum

        except Exception as netlist_error:
            error_msg = f"Error generating initial netlist: {str(netlist_error)}"
            app_logger.error(error_msg) # Replaced st.error
    else:
        app_logger.warning("No nodes found for netlist generation") # Replaced st.warning
        if active_results_dict.get('bboxes') is not None:
            try:
                app_logger.debug("Attempting to generate basic netlist from components only...")
                empty_nodes = []
                valueless_netlist_generated = current_analyzer.generate_netlist_from_nodes(empty_nodes, components_bboxes=active_results_dict['bboxes'])
                netlist_text_fallback = '\n'.join([current_analyzer.stringify_line(line) for line in valueless_netlist_generated])
                active_results_dict['netlist'] = valueless_netlist_generated
                active_results_dict['netlist_text'] = netlist_text_fallback
                active_results_dict['valueless_netlist_text'] = netlist_text_fallback
                # Removed: st.session_state.editable_netlist_content = netlist_text_fallback
                app_logger.info("Generated fallback netlist from components only.")
            except Exception as fallback_error:
                app_logger.error(f"Error generating fallback netlist: {str(fallback_error)}") # Replaced st.error

    detailed_timings_dict['Netlist Generation'] = time.time() - step_start_time_netlist
    return valueless_netlist_generated

def log_analysis_summary(active_results_dict, app_logger, configured_log_level_str): # Renamed param
    """Logs a summary of the analysis results if conditions are met."""
    # Convert string log level to logging module's level integer
    configured_log_level = getattr(logging, configured_log_level_str.upper(), logging.INFO)

    if active_results_dict.get('netlist') and app_logger.isEnabledFor(configured_log_level): # Check against actual level
        try:
            component_counts = {}
            for line in active_results_dict['netlist']:
                comp_type = line['class']
                if comp_type not in component_counts:
                    component_counts[comp_type] = 0
                component_counts[comp_type] += 1

            app_logger.info("Analysis results summary:")
            app_logger.info(f"- Image: {active_results_dict.get('uploaded_file_name', 'Unknown')}")
            app_logger.info(f"- Total components detected: {len(active_results_dict['netlist'])}")
            for comp_type, count in component_counts.items():
                app_logger.info(f"  - {comp_type}: {count}")
            if active_results_dict.get('nodes'):
                app_logger.info(f"- Total nodes: {len(active_results_dict['nodes'])}")
        except Exception as summary_error:
            app_logger.error(f"Error generating analysis summary: {str(summary_error)}")

def handle_final_netlist_generation(current_analyzer, active_results_dict, app_logger):
    """Handles the logic for the 'Get Final Netlist' button click, including VLM call.
    Modifies active_results_dict in place. Returns True on success, False on failure."""
    try:
        final_start_time = time.time()

        valueless_netlist = active_results_dict.get('netlist')
        enum_img = active_results_dict.get('enum_img')
        bbox_ids_for_fix = active_results_dict.get('bbox_ids')

        if not valueless_netlist or enum_img is None or not bbox_ids_for_fix:
            app_logger.error("Missing data for final netlist generation: valueless_netlist, enum_img, or bbox_ids.")
            return False # Indicate failure

        netlist = deepcopy(valueless_netlist)

        try:
            app_logger.debug("Calling Gemini for component labeling using (potentially cropped) enumerated image...")
            gemini_info = gemini_labels_openrouter(enum_img)
            app_logger.debug(f"Received information for {len(gemini_info) if gemini_info else 0} components from Gemini")
            app_logger.info(f"GEMINI BARE OUTPUT (gemini_info): {gemini_info}")
            active_results_dict['vlm_stage2_output'] = gemini_info

            current_analyzer.fix_netlist(netlist, gemini_info, bbox_ids_for_fix)
        except Exception as gemini_error:
            app_logger.error(f"Error calling VLM API: {str(gemini_error)}")
            app_logger.warning(f"Could not get component values from AI: {str(gemini_error)}. Using basic netlist.")
            # Fallback to valueless if VLM fails, but still proceed to stringify and store it
            # The netlist variable already holds the valueless_netlist if deepcopy happened before error

        if app_logger.isEnabledFor(logging.DEBUG):
            for i, ln_debug in enumerate(netlist):
                app_logger.debug(f"Netlist line {i} before stringify: {ln_debug}")

        netlist = [line for line in netlist if line.get('value') is not None and str(line.get('value')).strip().lower() != 'none']
        netlist_text = '\n'.join([current_analyzer.stringify_line(line) for line in netlist])

        active_results_dict['netlist'] = netlist
        active_results_dict['netlist_text'] = netlist_text
        # Removed: st.session_state.editable_netlist_content = netlist_text

        final_elapsed_time = time.time() - final_start_time
        if 'detailed_timings' in active_results_dict: # Ensure detailed_timings exists
            active_results_dict['detailed_timings']['Final Netlist Generation'] = final_elapsed_time
        else:
            active_results_dict['detailed_timings'] = {'Final Netlist Generation': final_elapsed_time}


        # Removed Streamlit specific calls:
        # st.session_state.final_netlist_generated = True
        # st.success("Final netlist generated successfully!")
        # st.rerun()
        app_logger.info("Final netlist generated successfully.")
        return True # Indicate success

    except Exception as e:
        app_logger.error(f"Error generating final netlist: {str(e)}") # Replaced st.error
        # Ensure netlist_text is at least the valueless one if an error occurs mid-process
        if 'valueless_netlist_text' in active_results_dict and 'netlist_text' not in active_results_dict:
            active_results_dict['netlist_text'] = active_results_dict['valueless_netlist_text']
        return False # Indicate failure
