"""Execution node - tool routing and execution logic"""
from typing import Dict, Any, List, Optional, Tuple
from state import AgentState
from pathlib import Path
import re


def execution_node(state: AgentState) -> Dict[str, Any]:
    """Perform feature extraction (including the segmentation step)"""
    feature_plan = state.get("feature_plan", {})
    image_paths = state.get("image_paths", [])
    analysis_results = state.get("analysis_results", {})
    segmentation_mask = state.get("segmentation_mask")
    
    if not feature_plan or "features" not in feature_plan:
        return {
            "analysis_results": analysis_results,
            "current_step": "execution",
        }
    
    features = feature_plan.get("features", [])
    
    # Separate features that need segmentation from those that do not
    features_needing_seg = [f for f in features if f.get("needs_segmentation", False)]
    features_no_seg = [f for f in features if not f.get("needs_segmentation", False)]
    
    # If some features need segmentation but there is no segmentation mask yet, perform segmentation first
    if features_needing_seg and segmentation_mask is None:
        print(f"\n[Execution] {len(features_needing_seg)} features require segmentation, executing segmentation...")
        segmentation_mask = _perform_segmentation(features_needing_seg, image_paths, state)
        if segmentation_mask is None:
            print("  ⚠️  Warning: Segmentation failed, skipping features that require segmentation")
            features_needing_seg = []
    
    # Note: VLM feature extraction is not performed in execution_node
    # VLM feature extraction is handled per sample in step 4 (batch feature extraction)
    # Here we only handle code features (if needed, but currently code features are also handled in step 4)
    
    # Separate code and VLM features (for statistics only)
    code_features_no_seg = [f for f in features_no_seg if f.get("method") == "code"]
    vlm_features_no_seg = [f for f in features_no_seg if f.get("method") == "vlm"]
    
    print(f"\n[Execution] Feature statistics: {len(code_features_no_seg)} code features (no seg), {len(vlm_features_no_seg)} VLM features (no seg)")
    print(f"  Note: All feature extraction (code and VLM) will be processed in batch in step 4")
    
    # If there are features requiring segmentation, also count them
    if features_needing_seg:
        code_features_seg = [f for f in features_needing_seg if f.get("method") == "code"]
        vlm_features_seg = [f for f in features_needing_seg if f.get("method") == "vlm"]
        print(f"  Requiring segmentation: {len(code_features_seg)} code features, {len(vlm_features_seg)} VLM features")
    
    return {
        "analysis_results": analysis_results,
        "segmentation_mask": segmentation_mask,
        "current_step": "execution",
    }


def _perform_segmentation(
    features: List[Dict[str, Any]],
    image_paths: List[str],
    state: AgentState
) -> Any:
    """Perform segmentation
    
    Args:
        features: list of features that need segmentation
        image_paths: list of image paths
        state: agent state
        
    Returns:
        The segmentation mask path (string), or None if it fails
    """
    from pathlib import Path
    from tools.segmentation import ensure_sample_segmentation, check_segmentation_exists
    
    if not image_paths:
        print(f"  [Segmentation] ⚠️  No image paths available, cannot perform segmentation")
        return None
    
    # Get sample_id from state to determine output path
    sample_id = state.get("sample_id", "")
    if not sample_id:
        print(f"  [Segmentation] ⚠️  Cannot determine sample ID, cannot perform segmentation")
        return None
    
    # Get data_root from state (needs to be inferred from image_paths)
    first_image_path = Path(image_paths[0])
    # Assume image_paths have the format data_root/sample_id/xxx.tif
    # We need to find data_root, here by searching upward from sample_id
    # But a simpler approach is to pass sample_dir in from the feature extraction pipeline
    
    # Temporary solution: infer sample_dir from image_paths[0]
    # Assume the path format: .../dataset/sample_id/image.tif
    sample_dir = first_image_path.parent
    
    # Check if segmentation result already exists
    existing_seg = check_segmentation_exists(sample_dir)
    if existing_seg:
        print(f"  [Segmentation] ✅ Found existing segmentation result: {existing_seg}")
        return str(existing_seg)
    
    # Collect segmentation parameters (extract from features if available)
    # Default to using the first image
    input_image_path = str(first_image_path)
    
    # Extract channel information from features (if available)
    channels = None
    for feature in features:
        seg_channels = feature.get("segmentation_channels")
        if seg_channels is not None:
            channels = seg_channels
            break
    
    # Execute segmentation with the configured backend (Allen by default)
    print(f"  [Segmentation] Starting segmentation: {first_image_path.name}")
    mask_path = ensure_sample_segmentation(
        sample_dir=sample_dir,
        image_path=input_image_path,
        channels=channels,
        conda_env=None,
    )
    
    if mask_path is not None:
        print(f"  [Segmentation] ✅ Segmentation completed: {mask_path}")
        return str(mask_path)
    else:
        print(f"  [Segmentation] ⚠️  Segmentation failed; continuing without masks")
        return None


def _extract_image_info_from_paths(image_paths: List[str]) -> Tuple[str, List[Dict[str, str]]]:
    """Extract channel/layer information from the image paths and generate an image list description
    
    Args:
        image_paths: list of image paths
        
    Returns:
        (image_list_description, image_info_list) tuple
        - image_list_description: image list description text
        - image_info_list: list of image info, where each element contains {index, filename, channel_or_layer, description}
    """
    if not image_paths:
        return "", []
    
    # Determine the data type: multi-channel 2D image (<=6 images) vs 3D image (>=6 images)
    num_images = len(image_paths)
    is_2d_multichannel = num_images <= 6
    is_3d = num_images >= 6
    
    # If there are exactly 6 images, prefer treating it as 2D multi-channel (since it is more common)
    if num_images == 6:
        is_2d_multichannel = True
        is_3d = False
    
    image_info_list = []
    
    for idx, img_path in enumerate(image_paths):
        path_obj = Path(img_path)
        filename = path_obj.name
        
        # Try to extract information from the filename
        # Format 1: slice_0000_Actin_F_actin.png (multi-channel 2D)
        # Format 2: slice_0000.png (3D)
        # Format 3: slice_0001_Tubulin_beta_tubulin.png (multi-channel 2D)
        
        channel_or_layer = None
        description = ""
        
        # Extract the slice number
        slice_match = re.search(r'slice[_\s]*(\d+)', filename, re.IGNORECASE)
        slice_num = int(slice_match.group(1)) if slice_match else idx
        
        if is_2d_multichannel:
            # Multi-channel 2D image: try to extract the channel name from the filename
            # Format 1: slice_0000_Actin_F_actin.png -> extract "Actin F actin"
            # Format 2: slice_0001_Tubulin_beta_tubulin.png -> extract "Tubulin beta tubulin"
            # Format 3: slice_0002_DAPI.png -> extract "DAPI"
            
            # Remove the extension
            name_without_ext = filename.rsplit('.', 1)[0]
            parts = name_without_ext.split('_')
            
            # Find the channel name after the slice and number parts
            channel_parts = []
            found_slice_num = False
            for i, part in enumerate(parts):
                # Look for the "slice" keyword (case-insensitive)
                if part.lower() == 'slice':
                    # The next part should be a number
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        found_slice_num = True
                        # Skip the slice and number, and take the following parts
                        if i + 2 < len(parts):
                            channel_parts = parts[i + 2:]
                        break
            
            if channel_parts:
                # Merge the channel name (e.g. ['Actin', 'F', 'actin'] -> 'Actin F actin')
                channel_name = ' '.join(channel_parts)
                channel_or_layer = channel_name
                description = f"Channel {idx} ({channel_name})"
            else:
                # If no channel name was found, use the index
                channel_or_layer = f"Channel {idx}"
                description = f"Channel {idx}"
        else:
            # 3D image: use the slice number as the layer number
            channel_or_layer = f"Z={slice_num}"
            description = f"Z-plane {slice_num} (layer {idx+1} of {num_images})"
        
        image_info_list.append({
            "index": idx + 1,
            "filename": filename,
            "channel_or_layer": channel_or_layer,
            "description": description
        })
    
    # Generate the image list description text
    if is_2d_multichannel:
        description_text = f"\n====================\nImage List (2D Multi-channel Data)\n====================\n"
        description_text += f"**Data Type**: 2D Multi-channel dataset with {num_images} channels.\n"
        description_text += f"**Important**: Each image represents a different biological marker (channel).\n\n"
        description_text += "The images you receive are in the following order:\n\n"
        for info in image_info_list:
            description_text += f"**Image {info['index']}**: `{info['filename']}`\n"
            description_text += f"  - Channel: {info['channel_or_layer']}\n"
            description_text += f"  - Description: {info['description']}\n\n"
        description_text += "**Critical**: When analyzing features, pay attention to which channel corresponds to which biological marker. "
        description_text += "Different features may be more visible in specific channels.\n"
    else:
        description_text = f"\n====================\nImage List (3D Volumetric Data)\n====================\n"
        description_text += f"**Data Type**: 3D volumetric dataset with {num_images} Z-planes.\n"
        description_text += f"**Important**: The images represent sequential Z-axis slices through a 3D volume.\n\n"
        description_text += "The images you receive are in the following order:\n\n"
        for info in image_info_list:
            description_text += f"**Image {info['index']}**: `{info['filename']}`\n"
            description_text += f"  - Z-plane: {info['channel_or_layer']}\n"
            description_text += f"  - Description: {info['description']}\n\n"
        description_text += "**Critical**: When analyzing features, consider the spatial distribution and patterns across the Z-axis. "
        description_text += "The sequence of images shows the progression through the 3D volume from bottom to top (or top to bottom).\n"
    
    return description_text, image_info_list


def _execute_code_feature(
    feature: Dict[str, Any],
    image_paths: List[str],
    state: AgentState,
    segmentation_mask: Any = None
) -> Any:
    """Perform code feature extraction
    
    Args:
        feature: feature definition
        image_paths: list of image paths
        state: agent state
        segmentation_mask: segmentation mask (if any)
        
    Returns:
        The feature value (for a single sample) or None
    """
    # Note: this function is now mainly used for executing on a single sample
    # Batch execution should use execute_feature_on_all_samples
    print(f"  [Code] Executing feature '{feature.get('name')}' (single sample)")
    print(f"    ⚠️  Note: For batch execution, please use execute_feature_on_all_samples")
    
    # Return None to indicate that batch execution is required
    return None


def _execute_vlm_feature(
    feature: Dict[str, Any],
    image_paths: List[str],
    state: AgentState,
    segmentation_mask: Any = None,
    log_file: Optional[str] = None
) -> Any:
    """Perform VLM feature scoring
    
    Args:
        feature: feature definition
        image_paths: list of image paths (should be PNG-format slices)
        state: agent state
        segmentation_mask: segmentation mask (if any; not yet implemented)
        
    Returns:
        The feature value (a float from 0-100)
    """
    from tools.vlm_client import get_vlm_client
    from nodes.prompt_gen import load_template, fill_template
    from config import settings
    from pathlib import Path
    
    feature_name = feature.get("name", "unknown")
    
    # Load VLM scoring template
    # Note: Template file name is vlm_scoring.json, but internal name is vlm_continuous_scoring_with_explanation
    try:
        template_dict = load_template("vlm_scoring")
        template_str = template_dict.get("template", "")
        if not template_str:
            raise ValueError("Template content is empty")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [VLM] ⚠️  Warning: Failed to load vlm_scoring template ({e}), using default prompt")
        template_str = "Evaluate feature {feature_name}: {feature_description}\nGive a score from 0-100."
    
    # Filter image paths to keep only VLM-supported formats (read from the config)
    from config import settings
    supported_extensions = settings.vlm_supported_formats
    vlm_paths = [p for p in image_paths if Path(p).suffix.lower() in supported_extensions]
    
    # Log detailed path information
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[VLM] Number of original image_paths: {len(image_paths)}\n")
            f.write(f"[VLM] Original image_paths: {[Path(p).name for p in image_paths[:5]]}\n")
            f.write(f"[VLM] Supported formats: {supported_extensions}\n")
            f.write(f"[VLM] Number of vlm_paths after filtering: {len(vlm_paths)}\n")
            if vlm_paths:
                f.write(f"[VLM] vlm_paths after filtering: {[Path(p).name for p in vlm_paths[:5]]}\n")
            f.flush()
    
    if not vlm_paths:
        # If there are no paths after filtering, check whether it is a slices directory issue
        # The slices directory may exist but the path selector returned .tif files
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM] ⚠️  Warning: No VLM-supported image files found ({supported_extensions})\n")
                f.write(f"[VLM] Trying to find PNG files in the slices directory...\n")
                f.flush()
        
        # Try to infer the slices directory from the first path in image_paths
        if image_paths:
            first_path = Path(image_paths[0])
            # If the path is a .tif file, try to find a slices subdirectory in the same directory
            if first_path.suffix.lower() == '.tif':
                sample_dir = first_path.parent
                slices_dir = sample_dir / "slices"
                if slices_dir.exists():
                    png_files = list(slices_dir.glob("*.png"))
                    if png_files:
                        vlm_paths = [str(p) for p in sorted(png_files)]
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"[VLM] ✅ Found {len(vlm_paths)} PNG files in the slices directory\n")
                                f.write(f"[VLM] Using files from the slices directory: {[Path(p).name for p in vlm_paths[:3]]}\n")
                                f.flush()
        
        # If still not found, use all paths (let the VLM client handle it)
        if not vlm_paths:
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM] ⚠️  Warning: Falling back to using all image_paths\n")
                    f.flush()
            vlm_paths = image_paths
    
    if not vlm_paths:
        error_msg = f"  [VLM] ❌ Error: No available image files (checked {len(image_paths)} paths, supported formats: {supported_extensions})"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.write(f"  [VLM] Original image_paths: {image_paths[:5]}{'...' if len(image_paths) > 5 else ''}\n")
                f.flush()
        return None
    
    # Log the start information
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[VLM] Starting feature evaluation for '{feature_name}'\n")
            f.write(f"[VLM] Using {len(vlm_paths)} image(s): {[Path(p).name for p in vlm_paths[:3]]}{'...' if len(vlm_paths) > 3 else ''}\n")
            f.flush()
    
    # Extract image information from the image paths and generate the image list description
    image_list_description, image_info_list = _extract_image_info_from_paths(vlm_paths)
    
    # Prepare the fill data
    temp_state = state.copy()
    temp_state.update({
        "feature_name": feature.get("name", ""),
        "feature_description": feature.get("description", ""),
        "feature_category": feature.get("category", ""),
        # Add image information
        "image_list_description": image_list_description,
        "image_info_list": image_info_list,
        "num_images_provided": len(vlm_paths),
    })
    
    # Fill the template (now includes image information)
    full_prompt = fill_template(template_str, temp_state)
    
    # Get the global VLM client (singleton, to avoid repeated loading)
    vlm_client = get_vlm_client()
    
    # Perform scoring (detailed information is written to the log; only brief info is printed here)
    import time
    start_time = time.time()
    try:
        score, full_response = vlm_client.score_feature(
            feature_def=feature,
            image_paths=vlm_paths,
            full_prompt=full_prompt,
            max_images=settings.vlm_max_images,  # Maximum number of images, read from the config
            log_file=log_file if log_file else None  # The log file passed in by the caller
        )
        
        elapsed = time.time() - start_time
        if elapsed > 600:  # More than 10 minutes
            print(f"  [VLM] ⚠️  Warning: VLM evaluation took {elapsed:.1f}s (>10 minutes)")
        
        return score
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        error_msg = f"  [VLM] ❌ Timeout after {elapsed:.1f}s: {e}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.flush()
        # Reload the VLM client to free memory
        from tools.vlm_client import _global_vlm_client
        import tools.vlm_client as vlm_module
        vlm_module._global_vlm_client = None
        # Re-raise the exception so the caller can catch it and retry
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"  [VLM] ❌ Error after {elapsed:.1f}s: {e}"
        print(error_msg)
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.write(f"Traceback:\n{tb_str}\n")
                f.flush()
        # If the error may be a memory issue, clean up the VLM client
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            import tools.vlm_client as vlm_module
            vlm_module._global_vlm_client = None
        return None


def _execute_vlm_features_batch(
    features: List[Dict[str, Any]],
    image_paths: List[str],
    state: AgentState,
    segmentation_mask: Any = None,
    log_file: Optional[str] = None,
    gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """Perform VLM scoring for multiple features in a batch (process all features in a single call)
    
    Args:
        features: list of feature definitions
        image_paths: list of image paths (should be PNG-format slices)
        state: agent state
        segmentation_mask: segmentation mask (if any; not yet implemented)
        log_file: log file path (optional)
        
    Returns:
        Dict where the key is the feature name and the value is the feature value (a float from 0-100)
    """
    from tools.vlm_client import get_vlm_client
    from nodes.prompt_gen import load_template, fill_template
    from config import settings
    from pathlib import Path
    
    if not features:
        return {}
    
    # Load VLM batch scoring template
    try:
        template_dict = load_template("vlm_scoring_batch")
        template_str = template_dict.get("template", "")
        if not template_str:
            raise ValueError("Template content is empty")
    except (FileNotFoundError, ValueError) as e:
        # If the batch template does not exist, fall back to the single-feature template, but the prompt needs modification
        print(f"  [VLM Batch] ⚠️  Warning: Failed to load vlm_scoring_batch template ({e}), using single template with batch prompt")
        try:
            template_dict = load_template("vlm_scoring")
            template_str = template_dict.get("template", "")
            # Modify the template to support multiple features
            # This will be handled in fill_template
        except:
            template_str = "Evaluate features: {features_list}\nGive scores from 0-100 for each feature."
    
    # Filter image paths to keep only VLM-supported formats (read from the config)
    supported_extensions = settings.vlm_supported_formats
    vlm_paths = [p for p in image_paths if Path(p).suffix.lower() in supported_extensions]
    
    # Log detailed path information
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[VLM Batch] Processing {len(features)} features\n")
            f.write(f"[VLM Batch] Feature list: {[feat.get('name') for feat in features]}\n")
            f.write(f"[VLM Batch] Number of original image_paths: {len(image_paths)}\n")
            f.write(f"[VLM Batch] Supported formats: {supported_extensions}\n")
            f.write(f"[VLM Batch] Number of vlm_paths after filtering: {len(vlm_paths)}\n")
            if vlm_paths:
                f.write(f"[VLM Batch] vlm_paths after filtering: {[Path(p).name for p in vlm_paths[:5]]}\n")
            f.flush()
    
    if not vlm_paths:
        # If there are no paths after filtering, check whether it is a slices directory issue
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM Batch] ⚠️  Warning: No VLM-supported image files found ({supported_extensions})\n")
                f.write(f"[VLM Batch] Trying to find PNG files in the slices directory...\n")
                f.flush()
        
        # Try to infer the slices directory from the first path in image_paths
        if image_paths:
            first_path = Path(image_paths[0])
            if first_path.suffix.lower() == '.tif':
                sample_dir = first_path.parent
                slices_dir = sample_dir / "slices"
                if slices_dir.exists():
                    png_files = list(slices_dir.glob("*.png"))
                    if png_files:
                        vlm_paths = [str(p) for p in sorted(png_files)]
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"[VLM Batch] ✅ Found {len(vlm_paths)} PNG files in the slices directory\n")
                                f.flush()
        
        # If still not found, use all paths
        if not vlm_paths:
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM Batch] ⚠️  Warning: Falling back to using all image_paths\n")
                    f.flush()
            vlm_paths = image_paths
    
    if not vlm_paths:
        error_msg = f"  [VLM Batch] ❌ Error: No available image files"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.flush()
        return {feat.get("name", "unknown"): None for feat in features}
    
    # Extract image information from the image paths and generate the image list description
    image_list_description, image_info_list = _extract_image_info_from_paths(vlm_paths)
    
    # Prepare the feature list information
    features_list = []
    for feature in features:
        features_list.append({
            "name": feature.get("name", "unknown"),
            "description": feature.get("description", ""),
            "category": feature.get("category", ""),
        })
    
    # Prepare the fill data
    temp_state = state.copy()
    temp_state.update({
        "features_list": features_list,
        "num_features": len(features),
        # Add image information
        "image_list_description": image_list_description,
        "image_info_list": image_info_list,
        "num_images_provided": len(vlm_paths),
    })
    
    # Fill the template (now includes image information)
    full_prompt = fill_template(template_str, temp_state)
    
    # Log the start information
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[VLM Batch] Starting batch evaluation for {len(features)} features\n")
            f.write(f"[VLM Batch] Using {len(vlm_paths)} image(s)\n")
            f.flush()
    
    # Get the VLM client (if gpu_id is specified, use the client for the corresponding GPU)
    vlm_client = get_vlm_client(gpu_id=gpu_id)
    
    # Perform batch scoring
    import time
    start_time = time.time()
    try:
        scores_dict, full_response = vlm_client.score_features_batch(
            features=features,
            image_paths=vlm_paths,
            full_prompt=full_prompt,
            max_images=settings.vlm_max_images,
            log_file=log_file if log_file else None
        )
        
        elapsed = time.time() - start_time
        if elapsed > 600:  # More than 10 minutes
            print(f"  [VLM Batch] ⚠️  Warning: VLM batch evaluation took {elapsed:.1f}s (>10 minutes)")
        
        return scores_dict
        
    except TimeoutError as e:
        elapsed = time.time() - start_time
        error_msg = f"  [VLM Batch] ❌ Timeout after {elapsed:.1f}s: {e}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.flush()
        # Reload the VLM client to free memory
        from tools.vlm_client import _global_vlm_client
        import tools.vlm_client as vlm_module
        vlm_module._global_vlm_client = None
        # Return a dict of None values
        return {feat.get("name", "unknown"): None for feat in features}
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"  [VLM Batch] ❌ Error after {elapsed:.1f}s: {e}"
        print(error_msg)
        import traceback
        tb_str = traceback.format_exc()
        print(tb_str)
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + "\n")
                f.write(f"Traceback:\n{tb_str}\n")
                f.flush()
        # If the error may be a memory issue, clean up the VLM client
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            import tools.vlm_client as vlm_module
            vlm_module._global_vlm_client = None
        return {feat.get("name", "unknown"): None for feat in features}


def _execute_single_feature(
    feature: Dict[str, Any],
    image_paths: List[str],
    segmentation_mask: Any = None,
    dataset_description: Optional[str] = None,
    expert_knowledge: Optional[str] = None,
    deep_research: Optional[str] = None,
    rag_knowledge: Optional[str] = None
) -> Any:
    """Execute a single feature (without depending on a full AgentState)
    
    Args:
        feature: feature definition
        image_paths: list of image paths
        segmentation_mask: segmentation mask (if any)
        dataset_description: dataset description (optional)
        expert_knowledge: expert knowledge (optional)
        
    Returns:
        The feature value
    """
    method = feature.get("method", "code")
    
    # Create a simplified state for execution
    state: AgentState = {
        "messages": [],
        "user_query": "",
        "sample_id": "",
        "image_paths": image_paths,
        "research_summary": dataset_description or "",
        "expert_examples": [],
        "expert_knowledge": expert_knowledge,
        "deep_research": deep_research,
        "rag_knowledge": rag_knowledge,
        "feature_plan": {"features": [feature]},
        "segmentation_mask": segmentation_mask,
        "analysis_results": {},
        "current_step": "execution",
        "iteration_count": 0,
        "error_log": [],
    }
    
    if method == "code":
        return _execute_code_feature(feature, image_paths, state, segmentation_mask)
    elif method == "vlm":
        return _execute_vlm_feature(feature, image_paths, state, segmentation_mask)
    else:
        print(f"  ⚠️  Unknown method: {method}")
        return None
