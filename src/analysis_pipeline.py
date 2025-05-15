import os
import shutil
import time
import logging
from copy import deepcopy
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
from src.utils import non_max_suppression_by_confidence, gemini_labels_openrouter, non_max_suppression_by_area

def process_new_upload(uploaded_file, upload_dir_path, app_logger):
    """Handles the processing of a newly uploaded image file."""
    app_logger.info(f"NEW IMAGE UPLOADED: {uploaded_file.name}")
    app_logger.info("="*80)

    # Convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    
    disk_image_path = os.path.join(upload_dir_path, uploaded_file.name)

    # Clear results when new image is uploaded
    st.session_state.active_results = {
        'bboxes': None,
        'nodes': None,
        'netlist': None,
        'netlist_text': None,
        'original_image': image, # Store the initially converted image
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
    
    # Store original image (already done above, but ensures it's set before EXIF processing)
    # st.session_state.active_results['original_image'] = image 
    
    # Clear and save files
    if os.path.exists(upload_dir_path):
        shutil.rmtree(upload_dir_path)
    os.makedirs(upload_dir_path, exist_ok=True)
    
    # Save using PIL to preserve EXIF data
    # Re-open from uploaded_file to get original bytes for PIL, as 'image' is already processed by cv2
    uploaded_file.seek(0) # Reset file pointer
    pil_image_for_exif = Image.open(uploaded_file)
    st.session_state.active_results['original_image_pil'] = pil_image_for_exif # STORE THE PIL IMAGE
        
    # Auto-rotate the image based on EXIF orientation tag
    app_logger.info(f"Checking EXIF orientation for {uploaded_file.name}")
    final_image_to_save = pil_image_for_exif # Start with the image PIL opened
    try:
        exif = pil_image_for_exif._getexif()
        if exif and 0x0112 in exif:  # 0x0112 is the orientation tag ID
            orientation = exif[0x0112]
            app_logger.info(f"Found orientation tag: {orientation}")
            if orientation != 1:  # If orientation is not normal
                app_logger.info(f"Auto-rotating image with orientation {orientation}")
                # Transpose the image PIL opened directly
                final_image_to_save = ImageOps.exif_transpose(pil_image_for_exif)
                # Update the numpy array image in session_state as well
                st.session_state.active_results['original_image'] = np.array(final_image_to_save)
    except Exception as e:
        app_logger.error(f"Error checking/rotating image based on EXIF: {str(e)}")
    
    # Save the processed image (final_image_to_save), preserving EXIF data if present on original
    # EXIF data should be extracted from the original PIL image, not the potentially transposed one
    # if we want to preserve original metadata. However, exif_transpose handles this.
    exif_bytes_to_save = final_image_to_save.info.get('exif')

    if exif_bytes_to_save:
        final_image_to_save.save(disk_image_path, format=final_image_to_save.format if final_image_to_save.format else uploaded_file.type.split('/')[-1].upper(), exif=exif_bytes_to_save)
        app_logger.info(f"Saved image {disk_image_path} with EXIF data ({len(exif_bytes_to_save)} bytes).")
    else:
        final_image_to_save.save(disk_image_path, format=final_image_to_save.format if final_image_to_save.format else uploaded_file.type.split('/')[-1].upper())
        app_logger.info(f"Saved image {disk_image_path} without EXIF data (EXIF not found or not preserved).")
    
    # Automatically trigger analysis
    st.session_state.start_analysis_triggered = True
    st.session_state.analysis_in_progress = True

def run_initial_detection_and_enrichment(current_analyzer, active_results, detailed_timings_dict, app_logger):
    """Performs initial component detection (YOLO) and semantic enrichment (LLaMA)."""
    step_start_time_yolo = time.time()
    raw_bboxes_orig_coords = []
    if active_results['original_image'] is not None:
        app_logger.debug("Step 1A: Detecting components with YOLO on original image...")
        raw_bboxes_orig_coords = current_analyzer.bboxes(active_results['original_image'])
        app_logger.debug(f"Detected {len(raw_bboxes_orig_coords)} raw bounding boxes on original image")
        
        bboxes_orig_coords_nms = non_max_suppression_by_confidence(raw_bboxes_orig_coords, iou_threshold=0.6)
        app_logger.debug(f"After NMS: {len(bboxes_orig_coords_nms)} bounding boxes on original image")
        active_results['bboxes_orig_coords_nms'] = bboxes_orig_coords_nms
    else:
        app_logger.error("No original image found for YOLO detection during analysis.")
        # This function is called within a try block, so an error here will be caught.
        raise ValueError("Original image not available for YOLO analysis.") 
    detailed_timings_dict['YOLO Component Detection'] = time.time() - step_start_time_yolo

    # --- NEW STEP: Reclassify terminals based on preliminary connectivity ---
    app_logger.info("Attempting preliminary reclassification of 'terminal' components...")
    if 'original_image_pil' in active_results and active_results['original_image_pil'] is not None:
        image_pil_for_reclass = active_results['original_image_pil']
        image_rgb_for_reclass = np.array(image_pil_for_reclass.convert("RGB"))
        current_analyzer.reclassify_terminals_based_on_connectivity(image_rgb_for_reclass, active_results['bboxes_orig_coords_nms']) # Modifies bboxes_orig_coords_nms in-place
        app_logger.info("Preliminary reclassification of 'terminal' components completed.")
        # active_results['bboxes_orig_coords_nms'] is now updated with reclassified classes
    else:
        app_logger.warning("Skipping terminal reclassification: 'original_image_pil' not found.")
    # --- END MOVED BLOCK ---

    # Store bboxes *after* reclassification but *before* LLaMA enrichment for debugging
    # This helps see what class LLaMA will receive.
    if active_results.get('bboxes_orig_coords_nms'):
         active_results['bboxes_after_reclass_before_llama'] = deepcopy(active_results['bboxes_orig_coords_nms'])

    # Step 1AA: Enrich BBoxes with Semantic Directions from LLaMA (Groq)
    # This step now operates on bboxes that have already been through terminal reclassification
    can_enrich = (
        active_results.get('bboxes_orig_coords_nms') and
        active_results.get('original_image') is not None and # original_image is numpy, used by _enrich_bboxes_with_directions
        hasattr(current_analyzer, 'groq_client') and current_analyzer.groq_client
    )
    if can_enrich:
        app_logger.info("Attempting to enrich component bboxes with semantic directions from LLaMA...")
        step_start_time_llama_enrich = time.time()
        try:
            current_analyzer._enrich_bboxes_with_directions(
                active_results['original_image'],
                active_results['bboxes_orig_coords_nms'] # This list is modified in-place
            )
            app_logger.info("Semantic direction enrichment step completed.")
        except Exception as e_enrich:
            app_logger.error(f"Error during LLaMA semantic direction enrichment: {e_enrich}")
            st.warning("Could not determine semantic directions for some components using LLaMA.")
        finally:
            detailed_timings_dict['LLaMA Direction Enrichment'] = time.time() - step_start_time_llama_enrich
    elif active_results.get('bboxes_orig_coords_nms') and active_results.get('original_image') is not None:
        app_logger.warning("Skipping LLaMA semantic direction enrichment: Groq client not available or other issue.")
    
    return active_results.get('bboxes_orig_coords_nms') # Return the (potentially reclassified and LLaMA-enriched) bboxes

def run_segmentation_and_cropping(current_analyzer, active_results, detailed_timings_dict, app_logger):
    """Performs YOLO-based cropping first, then SAM2 segmentation on the (potentially) cropped image."""
    original_image = active_results['original_image']
    # bboxes_orig_coords_nms are the YOLO detections on the original image, potentially LLaMA-enriched
    bboxes_from_initial_detection = active_results.get('bboxes_orig_coords_nms', [])

    # --- Step 1: YOLO-based Cropping ---
    step_start_time_yolo_crop = time.time()
    app_logger.debug("Attempting YOLO-based cropping...")
    image_after_yolo_crop, bboxes_after_yolo_crop, crop_debug_info = current_analyzer.crop_image_and_adjust_bboxes(
        original_image,
        deepcopy(bboxes_from_initial_detection), # Pass a copy to avoid modifying the original list if crop_image_and_adjust_bboxes does so
        padding=80 # Keep existing padding or adjust as needed
    )
    active_results['crop_debug_info'] = crop_debug_info
    detailed_timings_dict['YOLO-based Image Cropping'] = time.time() - step_start_time_yolo_crop

    if crop_debug_info and crop_debug_info.get('crop_applied'):
        app_logger.info(f"YOLO-based crop applied. New image dimensions: {image_after_yolo_crop.shape[:2]}")
        # The image_for_analysis will be this cropped version
        # bboxes_for_analysis will be bboxes_after_yolo_crop
    else:
        app_logger.info(f"YOLO-based crop NOT applied. Reason: {crop_debug_info.get('reason_for_no_crop', 'Unknown')}. Using original image dimensions.")
        # image_after_yolo_crop is original_image, bboxes_after_yolo_crop are bboxes_from_initial_detection
    
    # Update active_results with the image and bboxes that will be used for further analysis
    active_results['image_for_analysis'] = image_after_yolo_crop
    active_results['bboxes'] = bboxes_after_yolo_crop # These are adjusted to image_after_yolo_crop

    # --- Step 2: SAM2 Segmentation (on the potentially YOLO-cropped image) ---
    step_start_time_sam = time.time()
    binary_sam_mask_on_yolo_cropped_img = None
    sam2_colored_display_output = None

    if current_analyzer.use_sam2:
        app_logger.debug("Step 2B: Performing SAM2 segmentation on the (potentially YOLO-cropped) image_for_analysis...")
        # SAM2 expects BGR, but CircuitAnalyzer.segment_with_sam2 handles conversion if it receives RGB
        # image_after_yolo_crop is RGB, which is fine for segment_with_sam2
        binary_sam_mask_on_yolo_cropped_img, sam2_colored_display_output, _ = current_analyzer.segment_with_sam2(
            image_after_yolo_crop # Pass the (potentially) YOLO-cropped image
        )
        # The third return value (sam_extent_bbox from SAM2) is not critically needed here anymore for cropping decisions.
        active_results['sam2_output'] = sam2_colored_display_output # For display, now on yolo-cropped image
        if binary_sam_mask_on_yolo_cropped_img is not None:
            app_logger.info(f"SAM2 segmentation successful on YOLO-cropped image. Mask shape: {binary_sam_mask_on_yolo_cropped_img.shape}")
        else:
            app_logger.warning("SAM2 segmentation on YOLO-cropped image did not return a binary mask.")
    elif not current_analyzer.use_sam2:
        app_logger.warning("SAM2 is disabled. Skipping SAM2 segmentation.")
    
    detailed_timings_dict['SAM2 Segmentation on YOLO-Cropped Image'] = time.time() - step_start_time_sam
    
    # This will be used by node analysis. It's the SAM2 mask on the image_after_yolo_crop, or None.
    active_results['cropped_sam_mask_for_nodes'] = binary_sam_mask_on_yolo_cropped_img

    # The function returns the image that all subsequent analysis should use, 
    # the bboxes adjusted for that image, and the SAM2 mask (if any) for that image.
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
            app_logger.error(error_msg)
            st.error(error_msg)
            app_logger.warning("Continuing execution despite node analysis error")
            # nodes_identified remains None or its previous state
    elif not current_analyzer.use_sam2:
        app_logger.warning("Node analysis skipped: SAM2 is disabled.")
        st.warning("Node analysis cannot be performed as SAM2 is disabled.")
    else:
        app_logger.error("Node analysis skipped: Bounding boxes or cropped SAM2 mask not available.")
        st.error("Components or circuit mask not ready for node analysis.")
    
    detailed_timings_dict['Node Analysis'] = time.time() - step_start_time_nodes
    return nodes_identified # This will be active_results_dict['nodes']

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
            st.session_state.editable_netlist_content = valueless_netlist_text # For the editor
            
            # Generate initial netlist WITHOUT LLaMA directions for comparison
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
            app_logger.error(error_msg)
            st.error(error_msg)
            # valueless_netlist_generated remains None or its previous state
    else:
        app_logger.warning("No nodes found for netlist generation")
        st.warning("Could not identify connection nodes. The netlist may be incomplete.")
        if active_results_dict.get('bboxes') is not None: # bboxes here are bboxes_for_analysis
            try:
                app_logger.debug("Attempting to generate basic netlist from components only...")
                empty_nodes = [] # Effectively, generate from components if no connections
                valueless_netlist_generated = current_analyzer.generate_netlist_from_nodes(empty_nodes, components_bboxes=active_results_dict['bboxes'])
                netlist_text_fallback = '\n'.join([current_analyzer.stringify_line(line) for line in valueless_netlist_generated])
                active_results_dict['netlist'] = valueless_netlist_generated
                active_results_dict['netlist_text'] = netlist_text_fallback
                active_results_dict['valueless_netlist_text'] = netlist_text_fallback
                st.session_state.editable_netlist_content = netlist_text_fallback
                app_logger.info("Generated fallback netlist from components only.")
            except Exception as fallback_error:
                app_logger.error(f"Error generating fallback netlist: {str(fallback_error)}")
                st.error("Error generating fallback netlist from components.")

    detailed_timings_dict['Netlist Generation'] = time.time() - step_start_time_netlist
    return valueless_netlist_generated # This is active_results_dict['netlist']

def log_analysis_summary(active_results_dict, app_logger, configured_log_level):
    """Logs a summary of the analysis results if conditions are met."""
    if active_results_dict.get('netlist') and configured_log_level in ['DEBUG', 'INFO']:
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
    """Handles the logic for the 'Get Final Netlist' button click, including VLM call."""
    try:
        final_start_time = time.time()
        
        valueless_netlist = active_results_dict['netlist']
        enum_img = active_results_dict['enum_img']
        bbox_ids_for_fix = active_results_dict['bbox_ids'] 
        
        netlist = deepcopy(valueless_netlist)
        
        try:
            app_logger.debug("Calling Gemini for component labeling using (potentially cropped) enumerated image...")
            gemini_info = gemini_labels_openrouter(enum_img) 
            app_logger.debug(f"Received information for {len(gemini_info) if gemini_info else 0} components from Gemini")
            app_logger.info(f"GEMINI BARE OUTPUT (gemini_info): {gemini_info}")
            active_results_dict['vlm_stage2_output'] = gemini_info
            
            current_analyzer.fix_netlist(netlist, gemini_info, bbox_ids_for_fix)
        except Exception as gemini_error:
            app_logger.error(f"Error calling Gemini API: {str(gemini_error)}")
            st.warning(f"Could not get component values from AI: {str(gemini_error)}. Using basic netlist.")
            netlist = valueless_netlist # Fallback to valueless if VLM fails
        
        if app_logger.isEnabledFor(logging.DEBUG):
            for i, ln_debug in enumerate(netlist):
                app_logger.debug(f"App.py netlist line {i} before stringify: {ln_debug}")
        
        netlist = [line for line in netlist if line.get('value') is not None and str(line.get('value')).strip().lower() != 'none']
        netlist_text = '\n'.join([current_analyzer.stringify_line(line) for line in netlist])
        
        active_results_dict['netlist'] = netlist
        active_results_dict['netlist_text'] = netlist_text
        st.session_state.editable_netlist_content = netlist_text
        
        final_elapsed_time = time.time() - final_start_time
        if 'detailed_timings' in active_results_dict:
            active_results_dict['detailed_timings']['Final Netlist Generation'] = final_elapsed_time
        
        st.session_state.final_netlist_generated = True
        st.success("Final netlist generated successfully!")
        st.rerun()
            
    except Exception as e:
        st.error(f"Error generating final netlist: {str(e)}")
        app_logger.error(f"Error generating final netlist: {str(e)}")
