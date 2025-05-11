import os
import shutil
import time
import logging
from copy import deepcopy
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageOps
from src.utils import non_max_suppression_by_confidence, gemini_labels_openrouter

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

    # Step 1AA: Enrich BBoxes with Semantic Directions from LLaMA (Groq)
    can_enrich = (
        active_results.get('bboxes_orig_coords_nms') and
        active_results.get('original_image') is not None and
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
    
    return active_results.get('bboxes_orig_coords_nms') # Return the (potentially enriched) bboxes

def run_segmentation_and_cropping(current_analyzer, active_results, detailed_timings_dict, app_logger):
    """Performs SAM2 segmentation, determines extent, and crops the image and masks."""
    step_start_time_sam = time.time()
    full_binary_sam_mask, sam2_colored_display_output, sam_extent_bbox = None, None, None
    original_image = active_results['original_image']
    bboxes_orig_coords_nms = active_results.get('bboxes_orig_coords_nms', [])

    if original_image is not None and current_analyzer.use_sam2:
        app_logger.debug("Step 1B: Performing SAM2 segmentation on original image...")
        full_binary_sam_mask, sam2_colored_display_output, sam_extent_bbox = current_analyzer.segment_with_sam2(
            original_image
        )
        active_results['sam2_output'] = sam2_colored_display_output # For display
        if sam_extent_bbox:
            app_logger.info(f"SAM2 extent bbox found: {sam_extent_bbox}")
        else:
            app_logger.warning("SAM2 did not return a valid extent bbox. Cropping will be skipped.")
    elif not current_analyzer.use_sam2:
        app_logger.warning("SAM2 is disabled. Cropping based on SAM2 extent will be skipped.")
    detailed_timings_dict['SAM2 Segmentation & Extent'] = time.time() - step_start_time_sam

    # Step 1C: Crop based on SAM2 Extent
    step_start_time_crop = time.time()
    image_for_analysis = original_image # Default
    bboxes_for_analysis = bboxes_orig_coords_nms # Default
    cropped_sam_mask_for_nodes = full_binary_sam_mask # Default to full if no crop
    active_results['crop_debug_info'] = None # Initialize

    if sam_extent_bbox and full_binary_sam_mask is not None:
        app_logger.debug("Attempting to crop image and SAM2 mask based on SAM2 extent...")
        cropped_visual_image, adjusted_yolo_bboxes, crop_debug_info = current_analyzer.crop_image_and_adjust_bboxes(
            original_image,
            bboxes_orig_coords_nms,
            sam_extent_bbox,
            padding=80 
        )
        active_results['crop_debug_info'] = crop_debug_info

        if crop_debug_info and crop_debug_info.get('crop_applied'):
            image_for_analysis = cropped_visual_image
            bboxes_for_analysis = adjusted_yolo_bboxes
            final_crop_coords = crop_debug_info.get('final_crop_window_abs')
            if final_crop_coords and full_binary_sam_mask is not None:
                crop_x, crop_y, crop_x_max, crop_y_max = final_crop_coords
                cropped_sam_mask_for_nodes = full_binary_sam_mask[crop_y:crop_y_max, crop_x:crop_x_max]
                app_logger.info(f"Visual image and SAM2 mask cropped. Cropped SAM mask shape: {cropped_sam_mask_for_nodes.shape}")
            elif full_binary_sam_mask is None:
                 app_logger.warning("Full binary SAM mask is None, cannot crop it.")
            else:
                app_logger.warning("Final crop window coordinates not found. Cannot crop SAM mask.")
                cropped_sam_mask_for_nodes = full_binary_sam_mask 
        elif crop_debug_info:
            app_logger.info(f"Cropping was not applied. Reason: {crop_debug_info.get('reason_for_no_crop', 'Unknown')}. Using originals.")
        else:
            app_logger.warning("crop_debug_info was None or malformed. Using originals.")
    else:
        app_logger.info("Skipping crop: SAM2 extent bbox or full mask not available.")
    
    active_results['image_for_analysis'] = image_for_analysis
    active_results['bboxes'] = bboxes_for_analysis 
    active_results['cropped_sam_mask_for_nodes'] = cropped_sam_mask_for_nodes
    detailed_timings_dict['Image Cropping'] = time.time() - step_start_time_crop

    return image_for_analysis, bboxes_for_analysis, cropped_sam_mask_for_nodes

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
