"""MorphAgent main entry point - batch process the entire dataset"""
import sys
import re
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Queue, Process
import queue
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from tqdm import tqdm

from langchain_core.messages import HumanMessage

from config import settings, apply_api_provider
from graph import build_morph_agent_graph
from state import AgentState
from utils_helpers import find_image_paths, read_dataset_index, find_description_file, select_appropriate_data_source
from utils.cell_context import extract_cell_context
from knowledge.dataset_understanding import understand_dataset, get_dataset_description_text
from knowledge.expert_knowledge import extract_expert_knowledge
from knowledge.deep_research import extract_deep_research
from knowledge.rag import extract_rag_knowledge
from validation import ValidationExecutor


def _normalize_feature_key(name: str) -> str:
    """Convert a display name into the snake_case key commonly used in the merged code (lowercase, with spaces/slashes/parentheses etc. replaced by underscores)."""
    s = (name or "").lower().strip()
    s = re.sub(r"[\s/()]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _get_merged_feature_value(sample_features: Dict[str, Any], feature_display_name: str):
    """Retrieve a feature value by display name from the dict returned by extract_all after merging.
    The merged code commonly uses snake_case as keys, so here we first try the display name, then the normalized key, then a prefix match."""
    if not isinstance(sample_features, dict):
        return np.nan
    # 1) Exact match
    if feature_display_name in sample_features:
        return sample_features[feature_display_name]
    norm_display = _normalize_feature_key(feature_display_name)
    if not norm_display:
        return np.nan
    # 2) Normalized name as key
    if norm_display in sample_features:
        return sample_features[norm_display]
    # 3) Find a key such that: its normalized form equals the display name, or the display name is a prefix of the key (e.g. Nuc/Background -> nuc_background_ratio)
    candidates = []
    for k, v in sample_features.items():
        nk = _normalize_feature_key(k)
        if nk == norm_display:
            return v
        if norm_display.startswith(nk) or nk.startswith(norm_display):
            candidates.append((len(nk), k, v))
    if not candidates:
        return np.nan
    # Take the shortest matching key (e.g. nuc_background_ratio is preferred over longer ones)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2]


def _feature_name_for_method(name: str, method: str) -> str:
    """Normalize the feature name according to the enforced method: strip the vlm_ prefix for code, add the vlm_ prefix for vlm."""
    if not name or method not in ("code", "vlm"):
        return name
    if method == "code":
        return name[4:] if name.startswith("vlm_") else name
    # method == "vlm"
    return name if name.startswith("vlm_") else f"vlm_{name}"


def _extract_explicit_feature_specs(user_query: str) -> List[Dict[str, str]]:
    """Extract explicitly listed feature definitions from user_query.

    Only recognizes fragments of the form `1) feature_name: description`, and feature_name must be snake_case.
    This avoids mistakenly treating clauses like `(1) Two modalities: ...` as features.
    """
    if not user_query:
        return []

    # Require at least one underscore to reduce the chance of matching ordinary English phrases
    marker_re = re.compile(r"(\d+)\)\s*([a-zA-Z0-9]+_[a-zA-Z0-9_]+)\s*:\s*")
    matches = list(marker_re.finditer(user_query))
    if not matches:
        return []

    specs: List[Dict[str, str]] = []
    seen_names = set()
    for idx, m in enumerate(matches):
        name = m.group(2).strip()
        if not name or name in seen_names:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(user_query)
        desc = user_query[start:end].strip().strip(";").strip()
        if not desc:
            desc = f"Explicitly requested feature: {name}."
        specs.append({"name": name, "description": desc})
        seen_names.add(name)

    return specs


def _count_feature_columns(features_csv_path: Path) -> int:
    """Return the number of columns in the feature CSV other than sample_id."""
    if not features_csv_path.exists():
        return 0
    try:
        df = pd.read_csv(features_csv_path)
    except Exception:
        return 0
    return max(len([col for col in df.columns if col != "sample_id"]), 0)


def _load_validation_registry(registry_path: Path) -> Dict[str, Any]:
    """Load the validation registry; return an empty structure if it does not exist or is corrupted."""
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _build_basic_planner_feedback(
    feature_names: List[str],
    historical_feature_names: List[str],
) -> Dict[str, Any]:
    """Provide minimal compatible planner feedback when the validator is disabled."""
    return {
        "retained_features": [
            {"name": feature_name, "status": "unvalidated", "score": 0.0}
            for feature_name in feature_names
        ],
        "dropped_features": [],
        "redundancy_resolutions": [],
        "top_reason_codes": [],
        "all_historical_feature_names": historical_feature_names,
    }


def _detect_available_gpus() -> List[int]:
    """Detect available GPU devices
    
    Returns:
        List of available GPU IDs, e.g. [0, 1, 2, 3]
    """
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        else:
            return []
    except ImportError:
        # If torch is unavailable, try using nvidia-smi
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse the output to extract the number of GPUs
                lines = result.stdout.strip().split('\n')
                num_gpus = len(lines)
                return list(range(num_gpus))
        except:
            pass
        return []


def _vlm_worker_process(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    vlm_features: List[Dict[str, Any]],
    data_root,  # may be a Path or str
    feature_plan: Dict[str, Any],
    dataset_description: Optional[str],
    expert_knowledge: Optional[str],
    deep_research: Optional[str],
    rag_knowledge: Optional[str],
    round_results_dir,  # may be a Path or str
    needs_segmentation: bool
):
    """VLM worker process: handle the sample queue assigned to this GPU
    
    Args:
        gpu_id: GPU ID
        task_queue: task queue containing (sample_index, sample_id) tuples
        result_queue: result queue returning (sample_index, sample_id, results_dict) tuples
        vlm_features: list of VLM features
        data_root: dataset root directory
        feature_plan: feature plan
        dataset_description: dataset description
        expert_knowledge: expert knowledge
        deep_research: Deep Research knowledge
        rag_knowledge: RAG knowledge
        round_results_dir: results directory
        needs_segmentation: whether segmentation is needed
    """
    # Set up logging at the start of the function to capture all possible errors
    batch_log_file = None
    batch_log_fp = None
    
    # Convert argument types (spawn passes them as strings)
    try:
        if isinstance(data_root, str):
            data_root = Path(data_root)
        elif not isinstance(data_root, Path):
            data_root = Path(str(data_root))
        
        if round_results_dir:
            if isinstance(round_results_dir, str):
                round_results_dir = Path(round_results_dir)
            elif not isinstance(round_results_dir, Path):
                round_results_dir = Path(str(round_results_dir))
    except Exception as e:
        print(f"  ⚠️  GPU {gpu_id} worker process failed to convert argument types: {e}")
        return
    
    try:
        # Create a separate log file for each GPU
        if round_results_dir:
            batch_log_dir = round_results_dir / "features" / "vlm_batch" / f"gpu_{gpu_id}"
            batch_log_dir.mkdir(parents=True, exist_ok=True)
            batch_log_file = batch_log_dir / "execution_log.txt"
            batch_log_fp = open(batch_log_file, 'w', encoding='utf-8')
            batch_log_fp.write(f"[GPU {gpu_id}] ========== Worker process started ==========\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Process ID: {os.getpid()}\n")
            batch_log_fp.flush()
    except Exception as e:
        print(f"  ⚠️  GPU {gpu_id} failed to create log file: {e}")
        batch_log_fp = None
    
    try:
        # Import modules (done after the log file is created so import errors can be recorded)
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] Starting module import...\n")
            batch_log_fp.flush()
        
        from nodes.execution import _execute_vlm_features_batch
        from state import AgentState
        from utils_helpers import select_appropriate_data_source
        from tools.segmentation import check_segmentation_exists
        from tools.segmentation_tool import get_segmentation_tool
        
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] ✅ Modules imported successfully\n")
            batch_log_fp.flush()
    except Exception as e:
        error_msg = f"[GPU {gpu_id}] ❌ Module import failed: {e}"
        if batch_log_fp:
            batch_log_fp.write(f"{error_msg}\n")
            import traceback
            batch_log_fp.write(traceback.format_exc() + "\n")
            batch_log_fp.flush()
        print(f"  ⚠️  {error_msg}")
        if batch_log_fp:
            batch_log_fp.close()
        return
    
    try:
        # Note: in a multiprocessing environment, CUDA_VISIBLE_DEVICES should be set before the process starts
        # Setting it here may not take effect, so we specify the GPU via the device_map parameter in the VLM client
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)  # commented out, using device_map instead
        
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] Working directory: {os.getcwd()}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Data root directory: {data_root} (type: {type(data_root)})\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Whether data root directory exists: {Path(data_root).exists() if isinstance(data_root, (str, Path)) else 'N/A'}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Number of VLM features: {len(vlm_features)}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] VLM feature names: {[f.get('name') for f in vlm_features]}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Needs segmentation: {needs_segmentation}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Results directory: {round_results_dir}\n")
            batch_log_fp.write(f"[GPU {gpu_id}] ====================================\n")
            batch_log_fp.flush()
        
        # Process tasks in the queue
        task_count = 0
        
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] Starting to process queue tasks...\n")
            batch_log_fp.flush()
        
        while True:
            try:
                # Get a task from the queue (1-second timeout, used to check whether there are still tasks)
                task = None
                try:
                    task = task_queue.get(timeout=1)
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] ✅ Successfully got task: {task}\n")
                        batch_log_fp.flush()
                except queue.Empty:
                    # Queue is empty, keep waiting (other processes may still be working)
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] ⏳ Queue is empty, continuing to wait...\n")
                        batch_log_fp.flush()
                    continue
                except Exception as get_error:
                    # Some other error occurred while getting the task
                    error_msg = f"[GPU {gpu_id}] ❌ Error getting task from queue: {get_error}"
                    if batch_log_fp:
                        batch_log_fp.write(f"{error_msg}\n")
                        import traceback
                        batch_log_fp.write(traceback.format_exc() + "\n")
                        batch_log_fp.flush()
                    # Keep trying, do not exit
                    continue
                
                # Check whether the end signal (None) was received
                if task is None:  # end signal
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] Received end signal, exiting (processed {task_count} tasks)\n")
                        batch_log_fp.flush()
                    break
                
                task_count += 1
                try:
                    sample_index, sample_id = task
                except (ValueError, TypeError) as unpack_err:
                    error_msg = f"[GPU {gpu_id}] ❌ Task unpacking failed: task={task}, error={unpack_err}"
                    if batch_log_fp:
                        batch_log_fp.write(f"{error_msg}\n")
                        import traceback
                        batch_log_fp.write(traceback.format_exc() + "\n")
                        batch_log_fp.flush()
                    continue
                
                if batch_log_fp:
                    batch_log_fp.write(f"[GPU {gpu_id}] 📋 Starting to process task #{task_count}: sample_index={sample_index}, sample_id={sample_id}\n")
                    batch_log_fp.flush()
                
                if batch_log_fp:
                    batch_log_fp.write(f"\n{'='*60}\n")
                    batch_log_fp.write(f"[GPU {gpu_id}] Processing sample {sample_index}: {sample_id}\n")
                    batch_log_fp.write(f"Features: {[f.get('name') for f in vlm_features]}\n")
                    batch_log_fp.write(f"{'='*60}\n")
                    batch_log_fp.flush()
                
                # data_root was already converted to a Path object at the start of the function
                sample_dir = data_root / sample_id
                
                if batch_log_fp:
                    batch_log_fp.write(f"[GPU {gpu_id}] Sample directory path: {sample_dir}\n")
                    batch_log_fp.write(f"[GPU {gpu_id}] Whether sample directory exists: {sample_dir.exists()}\n")
                    batch_log_fp.flush()
                
                if not sample_dir.exists():
                    error_msg = f"Sample directory does not exist: {sample_dir}"
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] {sample_id}: ❌ {error_msg}\n")
                        batch_log_fp.flush()
                    result_queue.put((sample_index, sample_id, {}))
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                        batch_log_fp.flush()
                    continue
                
                # Intelligently select the data source
                if batch_log_fp:
                    batch_log_fp.write(f"[GPU {gpu_id}] Starting to select the data source...\n")
                    batch_log_fp.flush()
                
                try:
                    image_paths = select_appropriate_data_source(
                        sample_dir, 
                        vlm_features[0], 
                        dataset_description
                    )
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] Data source selection complete, found {len(image_paths) if image_paths else 0} image files\n")
                        if image_paths:
                            batch_log_fp.write(f"[GPU {gpu_id}] Image files: {image_paths[:3]}...\n")
                        batch_log_fp.flush()
                except Exception as e:
                    error_msg = f"Error selecting data source for sample {sample_id}: {e}"
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] {sample_id}: ❌ {error_msg}\n")
                        import traceback
                        batch_log_fp.write(traceback.format_exc() + "\n")
                        batch_log_fp.flush()
                    result_queue.put((sample_index, sample_id, {}))
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                        batch_log_fp.flush()
                    continue
                
                if not image_paths:
                    error_msg = f"No image files found for sample {sample_id}"
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] {sample_id}: ❌ {error_msg}\n")
                        batch_log_fp.flush()
                    result_queue.put((sample_index, sample_id, {}))
                    if batch_log_fp:
                        batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                        batch_log_fp.flush()
                    continue
                
                # Check and run segmentation (if needed)
                sample_seg_mask = None
                if needs_segmentation:
                    existing_seg = check_segmentation_exists(sample_dir)
                    if existing_seg:
                        sample_seg_mask = str(existing_seg)
                    else:
                        seg_tool = get_segmentation_tool()
                        channels = vlm_features[0].get("segmentation_channels")
                        seg_result = seg_tool.invoke({
                            "sample_dir": str(sample_dir),
                            "image_path": image_paths[0],
                            "channels": channels,
                            "conda_env": settings.segmentation_conda_env
                        })
                        if seg_result.get("success"):
                            sample_seg_mask = seg_result.get("mask_path")
                        else:
                            if batch_log_fp:
                                batch_log_fp.write(f"{sample_id}: ❌ Segmentation failed\n")
                            result_queue.put((sample_index, sample_id, {}))
                            continue
                
                # Build AgentState for batch VLM execution
                state: AgentState = {
                    "messages": [],
                    "user_query": "",
                    "sample_id": sample_id,
                    "image_paths": image_paths,
                    "research_summary": dataset_description or "",
                    "expert_examples": [],
                    "expert_knowledge": expert_knowledge,
                    "deep_research": deep_research,
                    "rag_knowledge": rag_knowledge,
                    "feature_plan": feature_plan,
                    "segmentation_mask": sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                    "analysis_results": {},
                    "current_step": "execution",
                    "iteration_count": 0,
                    "error_log": [],
                    # Batch feature information
                    "features_list": [{"name": f.get("name", ""), "description": f.get("description", ""), "category": f.get("category", "")} for f in vlm_features],
                    "num_features": len(vlm_features),
                }
                
                # Run batch VLM features (using the specified GPU)
                max_attempts = 3
                attempt_count = 0
                batch_results = None
                success = False
                
                while attempt_count < max_attempts and not success:
                    attempt_count += 1
                    try:
                        if attempt_count > 1:
                            retry_msg = f"[GPU {gpu_id}] {sample_id}: 🔄 Retry #{attempt_count - 1}"
                            if batch_log_fp:
                                batch_log_fp.write(f"{retry_msg}\n")
                                batch_log_fp.flush()
                        
                        # Use the VLM client for the specified GPU
                        if batch_log_fp:
                            batch_log_fp.write(f"[GPU {gpu_id}] Starting VLM batch processing call (gpu_id={gpu_id})\n")
                            batch_log_fp.write(f"[GPU {gpu_id}] Note: the model will be loaded onto GPU {gpu_id} on the first call to get_vlm_client\n")
                            batch_log_fp.flush()
                        
                        batch_results = _execute_vlm_features_batch(
                            vlm_features,
                            image_paths,
                            state,
                            segmentation_mask=sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                            log_file=str(batch_log_file) if batch_log_file else None,
                            gpu_id=gpu_id  # pass the GPU ID
                        )
                        
                        if batch_log_fp:
                            batch_log_fp.write(f"[GPU {gpu_id}] VLM batch processing returned: {batch_results is not None}\n")
                            if batch_results:
                                batch_log_fp.write(f"[GPU {gpu_id}] Number of result features: {len(batch_results)}\n")
                            batch_log_fp.flush()
                        
                        if batch_results:
                            has_valid_results = any(v is not None for v in batch_results.values())
                            if has_valid_results:
                                success = True
                            
                            if batch_log_fp:
                                if has_valid_results:
                                    batch_log_fp.write(f"{sample_id}: ✅ Batch scoring complete\n")
                                else:
                                    batch_log_fp.write(f"{sample_id}: ⚠️  Batch scoring complete but all results are None\n")
                                for feat_name, value in batch_results.items():
                                    batch_log_fp.write(f"  {feat_name}: {value}\n")
                                batch_log_fp.flush()
                            
                            # Put the result into the result queue
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] Preparing to put result into the result queue (sample_index={sample_index}, num_features={len(batch_results)})\n")
                                batch_log_fp.flush()
                            
                            result_queue.put((sample_index, sample_id, batch_results))
                            
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] ✅ Put result into the result queue\n")
                                batch_log_fp.flush()
                            break
                        else:
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] {sample_id}: ⚠️  Batch feature extraction returned empty results\n")
                                batch_log_fp.flush()
                            result_queue.put((sample_index, sample_id, {}))
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                                batch_log_fp.flush()
                            break
                    except TimeoutError as e:
                        error_msg = f"[GPU {gpu_id}] {sample_id}: ❌ Timeout error (attempt {attempt_count}/{max_attempts}): {e}"
                        if batch_log_fp:
                            batch_log_fp.write(f"{error_msg}\n")
                            batch_log_fp.flush()
                        
                        if attempt_count >= max_attempts:
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] {sample_id}: ❌ Timeout error, already tried {max_attempts} times, skipping this sample\n")
                                batch_log_fp.flush()
                            result_queue.put((sample_index, sample_id, {}))
                            if batch_log_fp:
                                batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                                batch_log_fp.flush()
                        else:
                            import time
                            time.sleep(1)
                    except Exception as e:
                        error_msg = f"[GPU {gpu_id}] {sample_id}: ❌ {e}"
                        if batch_log_fp:
                            batch_log_fp.write(f"{error_msg}\n")
                            import traceback
                            batch_log_fp.write(traceback.format_exc() + "\n")
                            batch_log_fp.flush()
                        result_queue.put((sample_index, sample_id, {}))
                        if batch_log_fp:
                            batch_log_fp.write(f"[GPU {gpu_id}] Put empty result into the result queue\n")
                            batch_log_fp.flush()
                        break
                
            except Exception as e:
                # Catch other unexpected errors
                error_msg = f"[GPU {gpu_id}] Error occurred while processing task: {e}"
                if batch_log_fp:
                    batch_log_fp.write(f"{error_msg}\n")
                    import traceback
                    batch_log_fp.write(traceback.format_exc() + "\n")
                    batch_log_fp.flush()
                # Continue with the next task, do not exit the process
                continue
            except KeyboardInterrupt:
                # Handle the interrupt signal
                if batch_log_fp:
                    batch_log_fp.write(f"[GPU {gpu_id}] Received interrupt signal, exiting\n")
                    batch_log_fp.flush()
                break
            except SystemExit:
                # Handle system exit
                if batch_log_fp:
                    batch_log_fp.write(f"[GPU {gpu_id}] Received system exit signal\n")
                    batch_log_fp.flush()
                raise  # re-raise so the system can exit normally
        
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] ========== Worker process finished ==========\n")
            batch_log_fp.write(f"[GPU {gpu_id}] Processed {task_count} tasks in total\n")
            batch_log_fp.flush()
    except Exception as outer_e:
        # Catch the outermost exception (including errors outside the loop)
        error_msg = f"[GPU {gpu_id}] Worker process encountered a serious error: {outer_e}"
        if batch_log_fp:
            batch_log_fp.write(f"{error_msg}\n")
            import traceback
            batch_log_fp.write(traceback.format_exc() + "\n")
            batch_log_fp.flush()
    finally:
        if batch_log_fp:
            batch_log_fp.write(f"[GPU {gpu_id}] Closing log file...\n")
            batch_log_fp.flush()
            batch_log_fp.close()


def process_sample(
    sample_id: str,
    data_root: Path,
    feature_plan: Dict[str, Any],
    app: Any,
    dataset_description: Optional[str] = None,
    expert_knowledge: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Process a single sample"""
    from nodes.execution import execution_node
    
    sample_dir = data_root / sample_id
    if not sample_dir.exists():
        print(f"⚠️  Sample directory does not exist: {sample_dir}")
        return None
    
    # Note: visualize_first_sample_channels uses a generic lookup and does not depend on features
    image_paths = find_image_paths(sample_dir, dataset_description)
    if not image_paths:
        print(f"⚠️  No image files found for sample {sample_id}")
        return None
    
    state: AgentState = {
        "messages": [],
        "user_query": "",
        "sample_id": sample_id,
        "image_paths": image_paths,
        "research_summary": dataset_description or "",
        "expert_examples": [],
        "expert_knowledge": expert_knowledge,
        "feature_plan": feature_plan,
        "segmentation_mask": None,
        "analysis_results": {},
        "current_step": "execution",
        "iteration_count": 0,
        "error_log": [],
    }
    
    result = execution_node(state)
    return result.get("analysis_results")


def visualize_first_sample_channels(
    first_sample_dir: Path,
    first_image_paths: List[str],
    results_dir: Path,
    dataset_description: Optional[str] = None
):
    """Save and visualize each channel of the first sample
    
    Args:
        first_sample_dir: directory path of the first sample
        first_image_paths: list of image file paths for the first sample
        results_dir: directory where results are saved
        dataset_description: dataset description (optional, used to determine channel information)
    """
    print(f"\nStep 2.5: Save and visualize each channel of the first sample")
    
    if not first_image_paths:
        print("  ⚠️  Warning: no image files found for the first sample, skipping visualization")
        return
    
    # Create the visualization directory
    viz_dir = results_dir / "first_sample_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to read the first image file
    first_image_path = Path(first_image_paths[0])
    if not first_image_path.exists():
        print(f"  ⚠️  Warning: image file does not exist: {first_image_path}")
        return
    
    try:
        # Read the image file (compatible with all formats: TIFF, PNG, JPG, etc.)
        print(f"  Reading image file: {first_image_path}")
        
        # Choose the reading method based on the file extension
        file_ext = first_image_path.suffix.lower()
        
        if file_ext in ['.tif', '.tiff']:
            # TIFF file: read with tifffile (supports multi-channel, multi-page)
            try:
                data = tifffile.imread(str(first_image_path))
            except Exception as e:
                # If tifffile fails, try reading with PIL
                print(f"  ⚠️  Warning: tifffile read failed, trying PIL: {e}")
                img = Image.open(str(first_image_path))
                data = np.array(img)
        else:
            # PNG, JPG, etc.: read with PIL
            img = Image.open(str(first_image_path))
            # Convert to a numpy array
            data = np.array(img)
            # If RGBA, convert to RGB
            if len(data.shape) == 3 and data.shape[2] == 4:
                # RGBA -> RGB
                data = data[:, :, :3]
        
        print(f"  Data shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        
        # Determine the number of channels
        if len(data.shape) == 3:
            # Shape is (C, H, W) or (H, W, C)
            if data.shape[0] < data.shape[2]:
                # Assume it is (C, H, W)
                num_channels = data.shape[0]
                height, width = data.shape[1], data.shape[2]
                channels = [data[i, :, :] for i in range(num_channels)]
            else:
                # Assume it is (H, W, C)
                num_channels = data.shape[2]
                height, width = data.shape[0], data.shape[1]
                channels = [data[:, :, i] for i in range(num_channels)]
        elif len(data.shape) == 2:
            # Single-channel image
            num_channels = 1
            height, width = data.shape
            channels = [data]
        else:
            print(f"  ⚠️  Warning: unsupported data shape: {data.shape}")
            return
        
        print(f"  Detected {num_channels} channels")
        print(f"  Image size: {height} x {width}")
        
        # Channel name mapping (automatically extracted from the dataset description)
        try:
            from utils_modules.channel_extractor import extract_channel_names_from_description
            channel_names = extract_channel_names_from_description(dataset_description, num_channels)
        except ImportError:
            # If the import fails, use default names
            channel_names = [f"Channel {i+1}" for i in range(num_channels)]
        
        # Save each channel
        print(f"  Saving each channel...")
        for i, (channel_data, channel_name) in enumerate(zip(channels, channel_names[:num_channels])):
            # Normalize to the 0-255 range (if the data is not already in this range)
            if channel_data.max() <= 1.0:
                channel_data_uint8 = (channel_data * 255).astype(np.uint8)
            else:
                channel_data_uint8 = np.clip(channel_data, 0, 255).astype(np.uint8)
            
            # Save as TIFF (clean special characters from the file name)
            # Replace all characters that could cause path problems
            safe_channel_name = channel_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            channel_filename = f"channel_{i+1:02d}_{safe_channel_name}.tif"
            channel_path = viz_dir / channel_filename
            
            # Ensure the directory exists
            channel_path.parent.mkdir(parents=True, exist_ok=True)
            
            tifffile.imwrite(str(channel_path), channel_data_uint8)
            print(f"    Saved: {channel_filename}")
            
            # Create a visualization image (using pseudocolor)
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original grayscale image
            im1 = axes[0].imshow(channel_data, cmap='gray')
            axes[0].set_title(f'Channel {i+1}: {channel_name}\n(Grayscale)', fontsize=12)
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0], fraction=0.046)
            
            # Pseudocolor image (using a different colormap)
            colormaps = ['hot', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm']
            cmap = colormaps[i % len(colormaps)]
            im2 = axes[1].imshow(channel_data, cmap=cmap)
            axes[1].set_title(f'Channel {i+1}: {channel_name}\n(Pseudocolor: {cmap})', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1], fraction=0.046)
            
            plt.tight_layout()
            
            # Save the visualization image (reuse the already cleaned safe_channel_name to avoid '/' or ':' being treated as path separators)
            viz_filename = f"channel_{i+1:02d}_{safe_channel_name}_visualization.png"
            viz_path = viz_dir / viz_filename
            plt.savefig(str(viz_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved visualization: {viz_filename}")
        
        # Create a summary figure of all channels
        print(f"  Creating summary figure of all channels...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (channel_data, channel_name) in enumerate(zip(channels, channel_names[:num_channels])):
            if i < len(axes):
                colormaps = ['hot', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm']
                cmap = colormaps[i % len(colormaps)]
                im = axes[i].imshow(channel_data, cmap=cmap)
                axes[i].set_title(f'Channel {i+1}: {channel_name}', fontsize=10)
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # Hide the extra subplots
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'All Channels - {first_sample_dir.name}', fontsize=14, y=0.995)
        plt.tight_layout()
        
        summary_path = viz_dir / "all_channels_summary.png"
        plt.savefig(str(summary_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved summary figure: all_channels_summary.png")
        
        print(f"  ✅ Visualization complete, files saved at: {viz_dir}")
        
    except Exception as e:
        print(f"  ❌ Error: an error occurred while processing the image: {e}")
        import traceback
        traceback.print_exc()


def execute_feature_on_all_samples(
    feature: Dict[str, Any],
    sample_ids: List[str],
    data_root: Path,
    feature_plan: Dict[str, Any],
    dataset_description: Optional[str] = None,
    expert_knowledge: Optional[str] = None,
    deep_research: Optional[str] = None,
    rag_knowledge: Optional[str] = None,
    results_dir: Optional[Path] = None,
    enable_critic: Optional[bool] = None,
    enable_segmentation: bool = True
) -> Dict[str, Any]:
    """Run a single feature on all samples
    
    Args:
        feature: feature definition
        sample_ids: list of all sample IDs
        data_root: dataset root directory
        feature_plan: feature plan
        dataset_description: dataset description
        expert_knowledge: expert knowledge
        deep_research: Deep Research knowledge
        rag_knowledge: RAG knowledge
        results_dir: directory where results are saved
        enable_segmentation: whether to enable data segmentation (default True); when False, seg is None
        
    Returns:
        Dictionary with sample_id as key and feature value as value
    """
    feature_name = feature.get("name", "unknown")
    feature_method = feature.get("method", "code")
    # Only segment when data segmentation is enabled and the feature requires it; otherwise the passed seg is None
    needs_segmentation_effective = feature.get("needs_segmentation", False) and enable_segmentation
    
    print(f"\n  Processing feature: {feature_name}")
    print(f"    Description: {feature.get('description', 'N/A')[:100]}...")
    print(f"    Method: {feature_method}")
    print(f"    Needs segmentation: {needs_segmentation_effective} (feature requirement={feature.get('needs_segmentation', False)}, segmentation switch={enable_segmentation})")
    
    # If the method is VLM, process samples one by one
    if feature_method == "vlm":
        from nodes.execution import _execute_vlm_feature
        from state import AgentState
        
        feature_results = {}
        segmentation_mask = None  # Initialize segmentation_mask variable
        
        # Only prepare the segmentation tool when segmentation is enabled and the feature requires it
        needs_segmentation = needs_segmentation_effective
        if needs_segmentation:
            from tools.segmentation import (
                check_segmentation_exists,
                segment_image_with_cellpose,
                get_segmentation_mask_path
            )
            from tools.segmentation_tool import get_segmentation_tool
            seg_tool = get_segmentation_tool()
            seg_prompt = feature.get("segmentation_prompt", "")
            channels = feature.get("segmentation_channels")
            print(f"    Segmentation prompt: {seg_prompt}")
            print(f"    Will check and run segmentation for each sample (if needed)")
        
        # Create the log file
        log_file = None
        if results_dir:
            log_dir = results_dir / "features" / feature_name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "execution_log.txt"
            log_fp = open(log_file, 'w', encoding='utf-8')
        else:
            log_fp = None
        
        try:
            # Process VLM features sample by sample (one VLM call per sample)
            # Use tqdm to show a progress bar and add detailed logging
            for sample_id in tqdm(sample_ids, desc="  VLM scoring", leave=False, ncols=80):
                # Record the currently processed sample to the log file
                if log_fp:
                    log_fp.write(f"\n{'='*60}\n")
                    log_fp.write(f"Processing sample: {sample_id}\n")
                    log_fp.write(f"{'='*60}\n")
                    log_fp.flush()  # ensure it is written immediately
                sample_dir = data_root / sample_id
                if not sample_dir.exists():
                    error_msg = f"Sample directory does not exist: {sample_dir}"
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    continue
                
                # Intelligently select the data source (based on the feature requirements)
                image_paths = select_appropriate_data_source(
                    sample_dir, 
                    feature, 
                    dataset_description
                )
                if not image_paths:
                    error_msg = f"No image files found for sample {sample_id}"
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    continue
                
                # For each sample, check and run segmentation (if needed)
                sample_seg_mask = None  # Initialize for each sample
                if needs_segmentation:
                    # Check whether a segmentation result already exists
                    existing_seg = check_segmentation_exists(sample_dir)
                    if existing_seg:
                        sample_seg_mask = str(existing_seg)
                        if log_fp:
                            log_fp.write(f"{sample_id}: ✅ Using existing segmentation result: {existing_seg}\n")
                    else:
                        # Run segmentation (using the LangChain tool)
                        print(f"    [Sample {sample_id}] Running segmentation...")
                        seg_result = seg_tool.invoke({
                            "sample_dir": str(sample_dir),
                            "image_path": image_paths[0],
                            "channels": channels,
                            "conda_env": settings.segmentation_conda_env
                        })
                        if seg_result.get("success"):
                            sample_seg_mask = seg_result.get("mask_path")
                            if log_fp:
                                log_fp.write(f"{sample_id}: ✅ Segmentation complete: {sample_seg_mask}\n")
                        else:
                            error_msg = f"Segmentation failed: {seg_result.get('message', 'Unknown error')}"
                            if log_fp:
                                log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                            print(f"    ⚠️  {sample_id}: {error_msg}")
                            continue  # skip this sample
                sample_dir = data_root / sample_id
                if not sample_dir.exists():
                    error_msg = f"Sample directory does not exist: {sample_dir}"
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    continue
                
                # Intelligently select the data source (based on the feature requirements)
                image_paths = select_appropriate_data_source(
                    sample_dir, 
                    feature, 
                    dataset_description
                )
                if not image_paths:
                    error_msg = f"No image files found for sample {sample_id}"
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    continue
                
                # Build AgentState for VLM execution
                state: AgentState = {
                    "messages": [],
                    "user_query": "",
                    "sample_id": sample_id,
                    "image_paths": image_paths,
                    "research_summary": dataset_description or "",
                    "expert_examples": [],
                    "expert_knowledge": expert_knowledge,
                    "deep_research": deep_research,
                    "rag_knowledge": rag_knowledge,
                    "feature_plan": feature_plan,
                    "segmentation_mask": sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                    "analysis_results": {},
                    "current_step": "execution",
                    "iteration_count": 0,
                    "error_log": [],
                    # Add feature information to state
                    "feature_name": feature.get("name", ""),
                    "feature_description": feature.get("description", ""),
                    "feature_category": feature.get("category", ""),
                }
                
                # Run the VLM feature (single sample, one VLM call)
                # Add retry logic: retry the same sample after a timeout, up to 3 attempts (1 initial + 2 retries)
                max_attempts = 3
                attempt_count = 0
                result = None
                success = False
                
                while attempt_count < max_attempts and not success:
                    attempt_count += 1
                    try:
                        if attempt_count > 1:
                            retry_msg = f"{sample_id}: 🔄 Retry #{attempt_count - 1} (of {max_attempts} attempts)"
                            print(f"    {retry_msg}")
                            if log_fp:
                                log_fp.write(f"{retry_msg}\n")
                                log_fp.flush()
                        
                        result = _execute_vlm_feature(
                            feature,
                            image_paths,
                            state,
                            segmentation_mask=sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                            log_file=log_file
                        )
                        if result is not None:
                            feature_results[sample_id] = result
                            success = True
                            if log_fp:
                                log_fp.write(f"{sample_id}: ✅ {result}\n")
                        else:
                            # If it returns None but was not a timeout, it may be another error; do not retry
                            if log_fp:
                                log_fp.write(f"{sample_id}: ⚠️  Feature extraction returned None\n")
                            break  # exit the retry loop
                    except TimeoutError as e:
                        error_msg = f"{sample_id}: ❌ Timeout error (attempt {attempt_count}/{max_attempts}): {e}"
                        if log_fp:
                            log_fp.write(f"{error_msg}\n")
                            log_fp.flush()
                        
                        if attempt_count >= max_attempts:
                            # Reached the maximum number of attempts, skip this sample
                            final_error = f"{sample_id}: ❌ Timeout error, already tried {max_attempts} times, skipping this sample"
                            print(f"    {final_error}")
                            if log_fp:
                                log_fp.write(f"{final_error}\n")
                                log_fp.flush()
                        else:
                            # Continue retrying, wait a short while before retrying
                            import time
                            time.sleep(1)  # wait 1 second before retrying
                    except Exception as e:
                        # Other exceptions are not retried, just skipped
                        error_msg = f"{sample_id}: ❌ {e}"
                        if log_fp:
                            log_fp.write(f"{error_msg}\n")
                        import traceback
                        if log_fp:
                            log_fp.write(traceback.format_exc() + "\n")
                        break  # exit the retry loop
        finally:
            if log_fp:
                log_fp.close()
                if log_file:
                    print(f"    Detailed results saved to: {log_file}")
        
        print(f"    Succeeded: {len(feature_results)}/{len(sample_ids)} samples")
        return feature_results
    
    # If the method is neither code nor vlm, use the old execution logic
    if feature_method not in ["code", "vlm"]:
        from nodes.execution import _execute_single_feature
        
        feature_results = {}
        segmentation_mask = None
        
        # If the feature needs segmentation, first run segmentation on one sample
        if feature.get("needs_segmentation", False):
            seg_prompt = feature.get("segmentation_prompt")
            if seg_prompt:
                print(f"    Segmentation prompt: {seg_prompt}")
                if sample_ids:
                    first_sample_dir = data_root / sample_ids[0]
                    first_image_paths = select_appropriate_data_source(
                        first_sample_dir, 
                        feature, 
                        dataset_description
                    )
                    if first_image_paths:
                        print(f"    Running segmentation on sample {sample_ids[0]}...")
                        # TODO: implement real segmentation
                        print(f"    ⚠️  Segmentation not yet implemented, skipping features that need seg")
                        return {}
        
        # Create the log file
        log_file = None
        if results_dir:
            log_dir = results_dir / "features" / feature_name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "execution_log.txt"
            log_fp = open(log_file, 'w', encoding='utf-8')
        else:
            log_fp = None
        
        try:
            # Run this feature on all samples
            for i, sample_id in enumerate(sample_ids, 1):
                sample_dir = data_root / sample_id
                if not sample_dir.exists():
                    error_msg = f"Sample directory does not exist: {sample_dir}"
                    if log_fp:
                        log_fp.write(f"[{i}/{len(sample_ids)}] {sample_id}: ❌ {error_msg}\n")
                    continue
                
                # Intelligently select the data source (based on the feature requirements)
                image_paths = select_appropriate_data_source(
                    sample_dir, 
                    feature, 
                    dataset_description
                )
                if not image_paths:
                    error_msg = f"No image files found for sample {sample_id}"
                    if log_fp:
                        log_fp.write(f"[{i}/{len(sample_ids)}] {sample_id}: ❌ {error_msg}\n")
                    continue
                
                # Run the feature
                try:
                    result = _execute_single_feature(
                        feature,
                        image_paths,
                        segmentation_mask=segmentation_mask,
                        dataset_description=dataset_description,
                        expert_knowledge=expert_knowledge,
                        deep_research=deep_research,
                        rag_knowledge=rag_knowledge
                    )
                    if result is not None:
                        feature_results[sample_id] = result
                        if log_fp:
                            log_fp.write(f"[{i}/{len(sample_ids)}] {sample_id}: ✅ {result}\n")
                    else:
                        if log_fp:
                            log_fp.write(f"[{i}/{len(sample_ids)}] {sample_id}: ⚠️  Feature extraction returned None\n")
                except Exception as e:
                    error_msg = str(e)
                    if log_fp:
                        log_fp.write(f"[{i}/{len(sample_ids)}] {sample_id}: ❌ {error_msg}\n")
                    import traceback
                    if log_fp:
                        log_fp.write(traceback.format_exc() + "\n")
        finally:
            if log_fp:
                log_fp.close()
                if log_file:
                    print(f"    Detailed results saved to: {log_file}")
        
        print(f"    Succeeded: {len(feature_results)}/{len(sample_ids)} samples")
        return feature_results
    
    # Code method: use the new code generation and execution module
    from tools.code_executor import run_code_generation_with_react
    from state import AgentState
    
    # Try to read mask order information from the segmentation summary
    segmentation_mask_order = ""
    try:
        if results_dir:
            segmentation_summary_path = results_dir / "segmentation_summary.json"
            if segmentation_summary_path.exists():
                import json
                with open(segmentation_summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    segmentation_mask_order = summary.get("mask_order_description", "")
    except Exception:
        pass
    
    # Build AgentState
    state: AgentState = {
        "messages": [],
        "user_query": "",
        "sample_id": sample_ids[0] if sample_ids else "",
        "image_paths": [],
        "research_summary": dataset_description or "",
        "expert_examples": [],
        "expert_knowledge": expert_knowledge,
        "deep_research": deep_research,
        "rag_knowledge": rag_knowledge,
        "feature_plan": feature_plan,
        "segmentation_mask": None,
        "analysis_results": {},
        "current_step": "execution",
        "iteration_count": 0,
        "error_log": [],
        # Add feature information to state
        "feature_name": feature.get("name", ""),
        "feature_description": feature.get("description", ""),
        "feature_category": feature.get("category", ""),
        # Add mask order information
        "segmentation_mask_order": segmentation_mask_order,
    }
    
    # Run code generation and execution (including ReAct logic)
    # Code features only use first-level files (raw data), not second-level files such as slices
    # Use DataPathSelector to ensure correct data source selection
    # Disable verbose output during batch processing and use a progress bar
    from tools.data_path_selector import get_data_path_selector
    selector = get_data_path_selector(verbose=False)  # disable verbose output during batch processing
    
    def find_code_data_sources(sample_dir, _):
        # Create a temporary feature dict with method "code"
        temp_feature = {"method": "code"}
        result = selector.select_data_paths(
            Path(sample_dir),
            temp_feature,
            dataset_description,
            method="code"
        )
        # Return format: {"image_paths": [...], "segmentation_paths": [...]}
        # For compatibility, return the list of image paths (the first image)
        if isinstance(result, dict):
            return result.get("image_paths", [])
        else:
            # Compatible with the old format
            return result if isinstance(result, list) else []
    
    # Print summary information once before batch processing
    if sample_ids:
        first_sample_dir = data_root / sample_ids[0]
        first_result = selector.select_data_paths(
            Path(first_sample_dir),
            {"method": "code"},
            dataset_description,
            method="code"
        )
        if isinstance(first_result, dict):
            first_image_paths = first_result.get("image_paths", [])
            first_seg_paths = first_result.get("segmentation_paths", [])
            if first_image_paths:
                print(f"  [Data Path Selector] Code features will use:")
                print(f"    Image file: {Path(first_image_paths[0]).name} ({len(sample_ids)} samples total)")
                if first_seg_paths:
                    print(f"    Segmentation files: {len(first_seg_paths)} ({', '.join([Path(p).name for p in first_seg_paths[:3]])}{'...' if len(first_seg_paths) > 3 else ''})")
    
    # Only check and run segmentation for each sample when segmentation is enabled and the feature requires it; otherwise seg is None during coding
    needs_segmentation = needs_segmentation_effective
    sample_paths_info = {}  # store path information for each sample
    
    if needs_segmentation:
        from tools.segmentation import (
            check_segmentation_exists,
            segment_image_with_cellpose,
            get_segmentation_mask_path
        )
        from tools.segmentation_tool import get_segmentation_tool
        seg_tool = get_segmentation_tool()
        channels = feature.get("segmentation_channels")
        
        print(f"    Checking and running segmentation for each sample...")
        for sample_id in sample_ids:
            sample_dir = data_root / sample_id
            if not sample_dir.exists():
                continue
            
            # Get the image path
            image_paths = find_code_data_sources(sample_dir, None)
            if not image_paths:
                continue
            
            image_path = Path(image_paths[0])
            
            # Check whether a segmentation result already exists
            existing_seg = check_segmentation_exists(sample_dir)
            if existing_seg:
                seg_mask_path = existing_seg
            else:
                # Run segmentation
                print(f"    [Sample {sample_id}] Running segmentation...")
                seg_result = seg_tool.invoke({
                    "sample_dir": str(sample_dir),
                    "image_path": str(image_path),
                    "channels": channels,
                    "conda_env": settings.segmentation_conda_env
                })
                if seg_result.get("success"):
                    seg_mask_path = Path(seg_result.get("mask_path"))
                else:
                    print(f"    ⚠️  {sample_id}: Segmentation failed")
                    continue
            
            # Compute the relative path (relative to data_root)
            rel_image_path = image_path.relative_to(data_root)
            rel_seg_mask_path = seg_mask_path.relative_to(data_root)
            
            sample_paths_info[sample_id] = {
                "image_path": str(rel_image_path),
                "segmentation_mask_path": str(rel_seg_mask_path),
                "absolute_image_path": str(image_path),
                "absolute_segmentation_mask_path": str(seg_mask_path)
            }
        
        print(f"    ✅ Completed segmentation check for {len(sample_paths_info)}/{len(sample_ids)} samples")
    
    # Add the path information to state for use by code generation
    if needs_segmentation:
        state["sample_paths_info"] = sample_paths_info
    
    extraction_result, code_result = run_code_generation_with_react(
        feature=feature,
        state=state,
        sample_ids=sample_ids,
        data_root=data_root,
        find_image_paths_func=find_code_data_sources,
        results_dir=results_dir,
        conda_env=None,  # use the default value from the config
        max_cycles=3,
        segmentation_mask_path=None,  # no longer use a single mask path, use sample_paths_info instead
        enable_critic=enable_critic  # pass the critic agent enabled state
    )
    
    # Return the result dictionary
    return extraction_result.values


def process_dataset_level_analysis(
    all_results: Dict[str, Any],
    data_root: Path
) -> Dict[str, Any]:
    """Dataset-level analysis (placeholder)"""
    # TODO: implement dataset-level analysis
    return {
        "summary": "Dataset-level analysis (to be implemented)",
        "total_samples": len(all_results),
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="MorphAgent - batch process the entire dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Dataset understanding: read the dataset description file and understand the dataset structure
  2. Feature planning: use the LLM to generate a feature plan (once, reused for all samples)
  3. Batch extraction: apply features to each sample (VLM/coding)
  4. Dataset analysis: aggregate results across all samples

Examples:
  python main.py "Generate unbiased features for 3D microscopy"
  python main.py "Generate features" --data-root /path/to/dataset
        """
    )
    
    parser.add_argument("user_query", type=str, help="User query")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the dataset root directory")
    parser.add_argument("--description", type=str, default=None, help="Path to the dataset description file")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory where results are saved")
    parser.add_argument("--enable-expert-knowledge", action="store_true", default=True, help="Enable expert knowledge extraction (enabled by default)")
    parser.add_argument("--disable-expert-knowledge", action="store_false", dest="enable_expert_knowledge", help="Disable expert knowledge extraction")
    parser.add_argument("--enable-deep-research", action="store_true", default=True, help="Enable Deep Research extraction (enabled by default)")
    parser.add_argument("--disable-deep-research", action="store_false", dest="enable_deep_research", help="Disable Deep Research extraction")
    parser.add_argument("--enable-rag", action="store_true", default=True, help="Enable RAG knowledge extraction (enabled by default)")
    parser.add_argument("--disable-rag", action="store_false", dest="enable_rag", help="Disable RAG knowledge extraction")
    parser.add_argument("--paddlex-device", type=str, default=None,
                        help="Device used by PaddleX to parse PDFs (cpu, gpu:0, ...). Defaults to config.paddlex_device (cpu).")

    # --- Auto Deep Research (single API call -> markdown report) --------------
    parser.add_argument("--auto-deep-research", action="store_true", default=False,
                        help="Before extraction, generate a deep-research report with ONE call to the deep-research model "
                             "(config.deep_research_model) and save it into the deep_research/ folder.")
    parser.add_argument("--deep-research-query", type=str, default=None,
                        help="Keywords/topic for --auto-deep-research (defaults to the user query).")

    # --- Auto Literature Retrieval (PubMed / Europe PMC -> RAG PDFs) ----------
    parser.add_argument("--auto-literature-retrieval", action="store_true", default=False,
                        help="Before RAG extraction, search PubMed/Europe PMC and download open-access PDFs into the RAG/ folder.")
    parser.add_argument("--pubmed-query", type=str, default=None,
                        help="Keywords for --auto-literature-retrieval (defaults to the user query).")
    parser.add_argument("--pubmed-max-results", type=int, default=None,
                        help="Max number of papers to download (defaults to config.pubmed_max_results).")
    parser.add_argument("--pubmed-min-year", type=int, default=None,
                        help="Only retrieve papers published in/after this year (defaults to config.pubmed_min_year).")
    parser.add_argument("--pubmed-include-non-oa", action="store_true", default=False,
                        help="Include non-open-access candidates in the search (PDF download may still fail for those).")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to the metadata CSV file (optional; used for metadata-aware validation when present)")
    parser.add_argument("--features-per-iteration", type=int, default=None, help="Number of features extracted per round (defaults to the value read from config.py, default 10)")
    parser.add_argument("--target-feature-count", type=int, default=None, help="Target total number of features (defaults to the value read from config.py, default 1000)")
    parser.add_argument("--num-rounds", type=int, default=1, help="Number of execution rounds (default 1 round)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume run: continue after the rounds already completed in --results-dir, without rerunning/overwriting completed rounds. "
                             "Must be used together with --results-dir pointing to an existing run directory.")
    parser.add_argument("--method", type=str, choices=["code", "vlm", "both"], default="both", 
                        help="Feature extraction method restriction: 'code'=use code generation only, 'vlm'=use VLM scoring only, 'both'=use all methods (default)")
    parser.add_argument("--temperature", type=float, default=0.0, 
                        help="Temperature parameter when the LLM generates features (default 0.0; higher values make the output more random. Suggested range: 0.0-1.0)")
    parser.add_argument("--reproduce", action="store_true", default=False,
                        help="Reproducible mode: temperature=0, fixed random seed, VLM uses deterministic decoding and enables cross-run score caching (in particular guarantees VLM reproducibility)")
    parser.add_argument("--reproduce-seed", type=int, default=42,
                        help="Global random seed used with --reproduce (default 42)")
    parser.add_argument("--multigpu", action="store_true", default=False,
                        help="Enable multi-GPU parallel processing of VLM features (one independent process per GPU card)")
    parser.add_argument("--code-parallel-workers", type=int, default=1,
                        help="Number of parallel processes for code feature extraction (default 1, i.e. serial processing; use parallel processing when set greater than 1)")
    parser.add_argument("--vlm-online-concurrency", type=int, default=1,
                        help="Number of concurrent threads for the online VLM API (effective when vlm-api-provider=online, default 1=serial; the API is I/O bound, so setting 8-16 can significantly speed things up)")
    
    # Illumination Correction parameters
    parser.add_argument("--enable-illumination-correction", action="store_true", default=None,
                        help="Enable Illumination Correction, based on the method of Singh et al. (J. Microscopy 2014)")
    parser.add_argument("--disable-illumination-correction", action="store_false", dest="enable_illumination_correction",
                        help="Disable illumination correction (default)")
    parser.add_argument("--illumination-correction-median-window", type=int, default=None,
                        help="Median filter window size for illumination correction (pixels, default 150, for 512x512 images)")
    parser.add_argument("--illumination-correction-downsample-factor", type=int, default=None,
                        help="Downsample factor for illumination correction (default 4, used to speed up ICF computation)")
    parser.add_argument("--illumination-correction-group-by-channel", action="store_true", default=None,
                        help="Compute ICF grouped by channel (default True, ICF computed separately per channel)")
    parser.add_argument("--illumination-correction-no-group-by-channel", action="store_false", dest="illumination_correction_group_by_channel",
                        help="Do not group by channel; compute ICF over all images together")
    
    # Feature extractor control parameters
    parser.add_argument("--code-vlm-ratio", type=float, default=0.5,
                        help="Ratio of Code to VLM features (float, should sum to 1.0, default 0.5 means 50%% each. For example: 0.7 means 70%% code, 30%% vlm)")
    parser.add_argument("--knowledge-dependency", type=float, default=0.5,
                        help="How much the feature extractor depends on existing knowledge (float, 0-1, default 0.5). 0=no dependence at all, freely exploring within the feature space; 1=fully dependent, no external expansion")
    parser.add_argument("--enable-background-knowledge-in-planning", action="store_true", default=True,
                        help="Enable background knowledge during features planning (enabled by default)")
    parser.add_argument("--disable-background-knowledge-in-planning", action="store_false", dest="enable_background_knowledge_in_planning",
                        help="Disable background knowledge during features planning (for ablation study)")
    parser.add_argument("--enable-critic-agent", action="store_true", default=None,
                        help="Enable the VLM critic agent to evaluate code (enabled by default, for ablation study)")
    parser.add_argument("--disable-critic-agent", action="store_false", dest="enable_critic_agent",
                        help="Disable the VLM critic agent for evaluating code (for ablation study)")
    
    # Data segmentation (cellpose-SAM) optional: enabled by default; when disabled the seg passed to the coding part is None
    parser.add_argument("--enable-segmentation", action="store_true", default=True,
                        help="Enable data segmentation (cellpose-SAM), enabled by default")
    parser.add_argument("--disable-segmentation", action="store_false", dest="enable_segmentation",
                        help="Disable data segmentation (do not run segmentation; seg is None during coding)")
    parser.add_argument("--segmentation-skip-if-present", action="store_true", default=True,
                        dest="segmentation_skip_if_present",
                        help="Skip cellpose if the sample already has any segmentation file (user uploads take priority, enabled by default)")
    parser.add_argument("--segmentation-run-even-if-present", action="store_false", dest="segmentation_skip_if_present",
                        help="Run cellpose even if the sample already has a segmentation file (skip only when the cellpose triplet is missing)")
    
    parser.add_argument("--api-provider", type=str, default="default",
                        help="LLM endpoint preset name (key in API_PROVIDER_PRESETS of config.py, case-insensitive). "
                             "Default 'default' uses LLM_BASE_URL/LLM_API_KEY/LLM_MODEL from environment variables / config.")
    parser.add_argument("--vlm-api-provider", type=str, default="online",
                        help="VLM provider: online/api=OpenAI-compatible multimodal API (default, recommended), scoring images one by one via the API; "
                             "qwen=locally self-hosted Qwen3-VL (advanced, requires installing GPU dependencies yourself).")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Override the LLM model name of the current api-provider preset (switch between different base models on the same gateway for comparison).")
    parser.add_argument("--vlm-online-model", type=str, default=None,
                        help="Override the multimodal model name used by the online VLM (--vlm-api-provider online). "
                             "Only effective when --vlm-api-provider is online/api.")
    parser.add_argument("--enable-feature-analysis", action="store_true", default=True,
                        dest="enable_feature_analysis",
                        help="Enable deterministic feature validation (enabled by default; automatically falls back to unsupervised validation when there is no metadata)")
    parser.add_argument("--disable-feature-analysis", action="store_false", dest="enable_feature_analysis",
                        help="Disable deterministic feature validation; the next round only passes already-extracted feature names to avoid duplicates")
    
    args = parser.parse_args()

    # --reproduce: enforce deterministic configuration (before provider / graph initialization)
    if args.reproduce:
        from utils_modules.reproducibility import apply_reproduce_mode
        settings.reproduce_mode = True
        settings.reproduce_seed = args.reproduce_seed
        settings.code_temperature = 0.0
        settings.vlm_temperature = 0.0
        args.temperature = 0.0
        if args.code_parallel_workers > 1:
            print(f"  [reproduce] code-parallel-workers {args.code_parallel_workers} -> 1 (deterministic)")
            args.code_parallel_workers = 1
        apply_reproduce_mode(args.reproduce_seed)
        print(f"  [reproduce] Enabled: temperature=0, seed={args.reproduce_seed}, VLM caching on")
    
    # Switch the LLM's base_url / api_key / model according to --api-provider
    apply_api_provider(args.api_provider)
    # Optional: override the preset LLM model name (switch base models on the same gateway for comparison)
    if args.llm_model:
        settings.llm_model = args.llm_model
        print(f"  Overriding LLM model: {settings.llm_model}")
    print(f"  Using API provider: {args.api_provider} (base_url={settings.llm_base_url}, model={settings.llm_model})")

    # Switch the VLM interface according to --vlm-api-provider (online/api=OpenAI-compatible multimodal API / qwen=local)
    from config import apply_vlm_provider
    apply_vlm_provider(args.vlm_api_provider)
    # Optional: override the online VLM multimodal model name
    if args.vlm_online_model:
        settings.vlm_online_model = args.vlm_online_model
        print(f"  Overriding online VLM model: {settings.vlm_online_model}")
    if settings.vlm_api_provider == "online":
        print(f"  Using VLM provider: online (base_url={settings.vlm_online_base_url}, model={settings.vlm_online_model})")
    else:
        print(f"  Using VLM provider: qwen (local Qwen3-VL: {settings.vlm_model_path})")
    
    # If command-line arguments were provided, override the default values in config
    # Illumination Correction parameters
    if args.enable_illumination_correction is not None:
        settings.enable_illumination_correction = args.enable_illumination_correction
        print(f"  Setting illumination correction: {'enabled' if settings.enable_illumination_correction else 'disabled'}")
    if args.illumination_correction_median_window is not None:
        settings.illumination_correction_median_window = args.illumination_correction_median_window
        print(f"  Setting illumination correction median filter window: {settings.illumination_correction_median_window}")
    if args.illumination_correction_downsample_factor is not None:
        settings.illumination_correction_downsample_factor = args.illumination_correction_downsample_factor
        print(f"  Setting illumination correction downsample factor: {settings.illumination_correction_downsample_factor}")
    if args.illumination_correction_group_by_channel is not None:
        settings.illumination_correction_group_by_channel = args.illumination_correction_group_by_channel
        print(f"  Setting illumination correction grouping mode: {'group by channel' if settings.illumination_correction_group_by_channel else 'all images together'}")
    
    if args.features_per_iteration is not None:
        settings.features_per_iteration = args.features_per_iteration
        print(f"  Setting number of features per round: {settings.features_per_iteration}")
    if args.target_feature_count is not None:
        settings.target_feature_count = args.target_feature_count
        print(f"  Setting target total number of features: {settings.target_feature_count}")
    
    # Critic Agent parameters
    if args.enable_critic_agent is not None:
        settings.enable_critic_agent = args.enable_critic_agent
        print(f"  Setting VLM Critic Agent: {'enabled' if settings.enable_critic_agent else 'disabled'}")
    
    # Data segmentation (cellpose-SAM) optional
    print(f"  Setting data segmentation: {'enabled' if args.enable_segmentation else 'disabled (seg is None during coding)'}")
    
    # Prepare data paths
    if args.data_root:
        input_data_root = Path(args.data_root).resolve()
    else:
        input_data_root = Path(settings.data_root)
        if not input_data_root.is_absolute():
            input_data_root = Path(__file__).parent / input_data_root
        input_data_root = input_data_root.resolve()
    
    # Intelligently detect the dataset path: if a dataset subdirectory exists, use it; otherwise use the input path itself
    if (input_data_root / "dataset").exists() and (input_data_root / "dataset").is_dir():
        data_root = input_data_root / "dataset"
        project_root = input_data_root  # project root directory (contains dataset and expert_knowledge)
    else:
        data_root = input_data_root
        project_root = input_data_root.parent  # assume dataset is under the project root directory
    
    # Create the results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        results_dir = data_root.parent / "results" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the description file
    description_path = find_description_file(
        data_root,
        Path(args.description) if args.description else None
    )
    
    print("="*60)
    print("🔬 MorphAgent - batch processing mode")
    print("="*60)
    print(f"Query: {args.user_query}")
    
    # Extract cell context information (single-cell vs multi-cell)
    cell_context = extract_cell_context(args.user_query)
    print(f"\nCell context detection:")
    print(f"  Type: {cell_context['cell_type']}")
    if cell_context['detection_keywords']:
        print(f"  Detected keywords: {', '.join(cell_context['detection_keywords'])}")
    else:
        print(f"  No explicit keywords detected, will use the generic mode")
    print(f"Project root directory: {project_root}")
    print(f"Dataset directory: {data_root}")
    if description_path:
        print(f"Dataset description file: {description_path}")
    print(f"Results directory: {results_dir}")

    if settings.reproduce_mode:
        cache_dir = Path(project_root) / ".morphagent_repro_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        settings.reproduce_cache_dir = str(cache_dir)
        print(f"  [reproduce] VLM score cache: {cache_dir}")
    
    # Step 1: Read the dataset index
    print(f"\nStep 1: Read the dataset index")
    sample_ids = read_dataset_index(data_root)
    print(f"  Found {len(sample_ids)} samples")
    if not sample_ids:
        print("❌ Error: no processable samples found")
        sys.exit(1)
    
    # Step 2: Dataset understanding
    print(f"\nStep 2: Dataset understanding")
    dataset_messages = understand_dataset(
        data_root,
        description_path,
        args.user_query
    )
    dataset_description = get_dataset_description_text(dataset_messages)
    
    # Step 2.1: Extract expert knowledge
    expert_knowledge = extract_expert_knowledge(
        project_root,
        enable_expert_knowledge=args.enable_expert_knowledge
    )
    
    # Save expert knowledge
    if expert_knowledge:
        expert_knowledge_path = results_dir / "expert_knowledge_summary.txt"
        with open(expert_knowledge_path, 'w', encoding='utf-8') as f:
            f.write(expert_knowledge)
        print(f"  Expert knowledge saved: {expert_knowledge_path}")
    
    # Resolve the PaddleX device (CLI overrides config; config default is cpu).
    paddlex_device = args.paddlex_device or settings.paddlex_device

    # Step 2.15: Auto Deep Research — generate a report with ONE API call and
    # drop it into the deep_research/ folder, so the extraction step below picks
    # it up. No local deep-research model is deployed.
    if getattr(args, "auto_deep_research", False):
        from knowledge.deep_research_agent import generate_deep_research_report
        dr_query = args.deep_research_query or args.user_query
        dr_dir = project_root / "deep_research"
        try:
            generate_deep_research_report(dr_query, dr_dir, dataset_description=dataset_description)
        except Exception as e:
            print(f"  [Deep Research] Auto report generation failed (continuing): {e}")

    # Step 2.2: Extract Deep Research
    deep_research = extract_deep_research(
        project_root,
        enable_deep_research=args.enable_deep_research,
        device=paddlex_device
    )
    
    # Save Deep Research
    if deep_research:
        deep_research_path = results_dir / "deep_research_summary.txt"
        with open(deep_research_path, 'w', encoding='utf-8') as f:
            f.write(deep_research)
        print(f"  Deep Research saved: {deep_research_path}")
    
    # Step 2.25: Auto Literature Retrieval — search PubMed/Europe PMC and
    # download open-access PDFs into the RAG/ folder so extract_rag_knowledge
    # (PaddleX -> LLM summary) can consume them.
    if getattr(args, "auto_literature_retrieval", False) and args.enable_rag:
        from knowledge.pubmed_fetcher import fetch_pubmed_literature
        lit_query = args.pubmed_query or args.user_query
        rag_dir = project_root / "RAG"
        max_results = args.pubmed_max_results if args.pubmed_max_results is not None else settings.pubmed_max_results
        min_year = args.pubmed_min_year if args.pubmed_min_year is not None else settings.pubmed_min_year
        try:
            fetch_pubmed_literature(
                lit_query, rag_dir,
                max_results=max_results,
                min_year=min_year,
                open_access_only=not args.pubmed_include_non_oa,
                email=settings.ncbi_email,
                api_key=settings.ncbi_api_key,
            )
        except Exception as e:
            print(f"  [Literature Retrieval] Failed (continuing with any existing PDFs): {e}")

    # Step 2.3: Extract RAG knowledge
    rag_knowledge = extract_rag_knowledge(
        project_root,
        enable_rag=args.enable_rag,
        device=paddlex_device
    )
    
    # Save RAG knowledge
    if rag_knowledge:
        rag_knowledge_path = results_dir / "rag_knowledge_summary.txt"
        with open(rag_knowledge_path, 'w', encoding='utf-8') as f:
            f.write(rag_knowledge)
        print(f"  RAG knowledge saved: {rag_knowledge_path}")
    
    # Step 2.4: Segment all samples (cell body, nucleus, plastid) -- optional, can be turned off with --disable-segmentation
    print(f"\nStep 2.4: Data segmentation ({'running' if args.enable_segmentation else 'disabled, seg is None during coding'})")
    segmentation_mask_order = ""
    segmentation_results = {}
    
    if args.enable_segmentation:
        from tools.segmentation import segment_all_samples, list_segmentation_files
        from tools.data_statistics import _generate_mask_order_description
        
        # Channels used for segmentation. Default None -> let the Cellpose-SAM
        # backend auto-infer sensible channels (single-channel used directly;
        # 3-channel defaults to cyto=[0,2], nuclei=[2]). Datasets with a
        # non-standard channel order can set an explicit list here.
        segmentation_channels = None
        
        # Run batch segmentation (optionally skipped when the user already has any segmentation file)
        segmentation_results = segment_all_samples(
            sample_ids=sample_ids,
            data_root=data_root,
            dataset_description=dataset_description,
            channels=segmentation_channels,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            tile_norm_blocksize=0,
            batch_size=32,
            conda_env=settings.segmentation_conda_env,
            skip_if_any_segmentation_exists=getattr(args, "segmentation_skip_if_present", True)
        )
        
        # Tally the segmentation results (success / skipped_user_seg / failed)
        success_count = sum(1 for v in segmentation_results.values() if v == "success")
        skipped_count = sum(1 for v in segmentation_results.values() if v == "skipped_user_seg")
        print(f"  Segmentation stats: {success_count} succeeded, {skipped_count} skipped (already segmented), {len(sample_ids) - success_count - skipped_count} failed")
        
        # Get the mask order from the first sample that has any segmentation file (consistent with data_path_selector, including user uploads)
        first_sample_with_seg = None
        for sample_id in sample_ids:
            sample_dir = data_root / sample_id
            files = list_segmentation_files(sample_dir)
            if files:
                first_sample_with_seg = sample_id
                break
        if first_sample_with_seg:
            sample_dir = data_root / first_sample_with_seg
            files = list_segmentation_files(sample_dir)
            seg_files_info = []
            for idx, (name, path) in enumerate(files, 1):
                if path.exists():
                    seg_files_info.append({"index": idx, "name": name, "stem": path.stem})
            if seg_files_info:
                segmentation_mask_order = _generate_mask_order_description(seg_files_info)
                print(f"  ✅ Generated mask order description (based on sample {first_sample_with_seg}, {len(seg_files_info)} files total)")
    else:
        # Segmentation not enabled: record everything as "not run"; during coding data_path_selector will not return seg, i.e. seg is None/empty
        segmentation_results = {sid: "failed" for sid in sample_ids}
        print(f"  Skipped segmentation ({len(sample_ids)} samples total)")
    
    # Save the segmentation results summary (includes mask order information; mask_order_description is empty when segmentation is disabled)
    segmentation_summary_path = results_dir / "segmentation_summary.json"
    import json
    success_count = sum(1 for v in segmentation_results.values() if v == "success")
    skipped_count = sum(1 for v in segmentation_results.values() if v == "skipped_user_seg")
    failed_count = len(sample_ids) - success_count - skipped_count
    # Defensively create the directory: avoid the results directory momentarily missing due to external cleanup/concurrency
    segmentation_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(segmentation_summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(sample_ids),
            "successful": success_count,
            "skipped_user_seg": skipped_count,
            "failed": failed_count,
            "results": segmentation_results,
            "mask_order_description": segmentation_mask_order,
            "segmentation_enabled": args.enable_segmentation
        }, f, indent=2, ensure_ascii=False)
    print(f"  Segmentation results summary saved: {segmentation_summary_path}")
    
    # Step 2.5: Save and visualize each channel of the first sample
    first_sample_dir = data_root / sample_ids[0]
    # Visualization uses a generic lookup and does not depend on features
    first_image_paths = find_image_paths(first_sample_dir, dataset_description)
    visualize_first_sample_channels(
        first_sample_dir,
        first_image_paths,
        results_dir,
        dataset_description
    )
    
    # Save the mask order information into state (for subsequent code generation)
    # Note: state is used in subsequent steps; here we ensure the information is available
    if segmentation_mask_order:
        # state will be built in subsequent steps; here we first save it to a global variable or pass it in another way
        # In fact, state is built in execute_feature_on_all_samples, and we need to pass it there
        # But to make the information flow robust, we can also save it to a file here and read it when needed
        pass  # the information is already saved in segmentation_summary.json and will be read from there during code generation
    
    # Initialize variables related to multi-round execution
    print(f"  [config] LLM Temperature: {args.temperature}")
    app = build_morph_agent_graph(temperature=args.temperature)
    cumulative_analysis_results = []  # store the analysis results of all rounds
    previous_features_summary = None  # feature summary of the previous round
    validation_executor = ValidationExecutor()
    
    # Define the cumulative feature CSV file paths (defined outside the loop, shared by all rounds)
    features_csv_path = results_dir / "features.csv"
    retained_features_csv_path = results_dir / "retained_features.csv"
    feature_registry_path = results_dir / "feature_registry.json"

    # Resume run: scan the completed rounds (round_results.json being written marks completion) and continue from max+1
    start_round = 1
    if args.resume:
        completed_rounds = []
        for round_dir in results_dir.glob("round_*"):
            if not round_dir.is_dir():
                continue
            match = re.search(r"round_(\d+)$", round_dir.name)
            if match and (round_dir / "round_results.json").exists():
                completed_rounds.append(int(match.group(1)))
        if completed_rounds:
            last_completed = max(completed_rounds)
            start_round = last_completed + 1
            print(f"  ♻️  Resume run: detected {len(completed_rounds)} completed rounds (highest round {last_completed}), continuing from round {start_round}")
            if features_csv_path.exists():
                try:
                    _resume_df = pd.read_csv(features_csv_path)
                    print(f"     Features accumulated so far: {len([c for c in _resume_df.columns if c != 'sample_id'])} (from {features_csv_path.name})")
                except Exception:
                    pass
        else:
            print(f"  ♻️  Resume run: no completed rounds found in {results_dir}, starting from round 1")
        if start_round > args.num_rounds:
            print(f"  ✅ The completed rounds ({start_round - 1}) already meet or exceed the target number of rounds ({args.num_rounds}), no need to continue")

    # Multi-round execution loop
    for round_num in range(start_round, args.num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"🔄 Feature extraction round {round_num}/{args.num_rounds}")
        print(f"{'='*80}")
        
        # Create the results directory for this round
        round_results_dir = results_dir / f"round_{round_num}"
        round_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Feature planning (using the first sample)
        print(f"\nStep 3: Feature planning (using sample {sample_ids[0]})")
        
        # Prepare the analysis results summary (to be passed to feature planning)
        analysis_summary_for_planning = None
        prior_registry = _load_validation_registry(feature_registry_path)
        
        # Read all existing feature names in the current CSV (used to avoid duplicates)
        existing_feature_names = []
        existing_feature_names = prior_registry.get("all_historical_feature_names", []) or []
        if not existing_feature_names and features_csv_path.exists():
            try:
                existing_df = pd.read_csv(features_csv_path)
                existing_feature_names = [col for col in existing_df.columns if col != 'sample_id']
            except:
                pass
        
        if round_num > 1 and previous_features_summary:
            # Build the analysis results summary
            analysis_summary_for_planning = {
                "previous_rounds": round_num - 1,
                "previous_features_summary": previous_features_summary,
                "cumulative_analysis": cumulative_analysis_results[-1] if cumulative_analysis_results else None,
                "all_existing_feature_names": existing_feature_names  # add all existing feature names
            }
        elif existing_feature_names:
            # Even without analysis results, pass the existing feature names (after the first round)
            analysis_summary_for_planning = {
                "previous_rounds": round_num - 1,
                "all_existing_feature_names": existing_feature_names
            }
        
        # Set the available methods based on the arguments
        if args.method == "code":
            available_methods = "code"
            method_instructions = "Important: you may only choose the 'code' method. All features must be implemented using code generation."
        elif args.method == "vlm":
            available_methods = "vlm"
            method_instructions = "Important: you may only choose the 'vlm' method. All features must be implemented using VLM scoring."
        else:  # both
            available_methods = "code, vlm"
            method_instructions = "None"
        
        # Try to read mask order information from the segmentation summary
        segmentation_mask_order_from_file = ""
        try:
            # Build the path from results_dir (segmentation_summary.json is saved in the results root directory)
            if results_dir:
                seg_summary_path = results_dir / "segmentation_summary.json"
                if seg_summary_path.exists():
                    import json
                    with open(seg_summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        segmentation_mask_order_from_file = summary.get("mask_order_description", "")
        except Exception as e:
            # Fail silently, does not affect the main flow
            pass
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=args.user_query)],
            "user_query": args.user_query,
            "sample_id": sample_ids[0],
            "image_paths": first_image_paths,
            "research_summary": dataset_description,
            "expert_examples": [],
            "expert_knowledge": expert_knowledge,
            "deep_research": deep_research,
            "rag_knowledge": rag_knowledge,
            "feature_plan": None,
            "segmentation_mask": None,
            "analysis_results": {},
            "current_step": "start",
            "iteration_count": 0,
            "error_log": [],
            # Add fields related to multi-round execution
            "round_number": round_num,
            "previous_analysis_summary": analysis_summary_for_planning,
            # Add method restriction
            "available_methods": available_methods,
            "method_instructions": method_instructions,
            # Add mask order information
            "segmentation_mask_order": segmentation_mask_order_from_file,
            # Add feature extractor control parameters
            "code_vlm_ratio": args.code_vlm_ratio,
            "knowledge_dependency": args.knowledge_dependency,
            # Add background knowledge control parameter (for ablation study)
            "enable_background_knowledge_in_planning": args.enable_background_knowledge_in_planning,
            # Whether to enable deterministic feature validation (used to inject the Validation Design Spec into the planning prompt)
            "enable_feature_analysis": args.enable_feature_analysis,
        }
        
        # Run the graph to get the feature plan (in reproduce mode, reuse the
        # cross-run canonical feature_plan only when it matches the requested scale).
        feature_plan = None
        plan_cache_path = None
        if settings.reproduce_mode and settings.reproduce_cache_dir and round_num == 1:
            requested_n = int(settings.features_per_iteration)
            requested_t = int(settings.target_feature_count)
            plan_cache_path = (
                Path(settings.reproduce_cache_dir)
                / f"feature_plan_{args.method}_n{requested_n}_t{requested_t}.json"
            )
            legacy_plan_cache = Path(settings.reproduce_cache_dir) / f"feature_plan_{args.method}.json"
            for candidate in (plan_cache_path, legacy_plan_cache):
                if not candidate.exists():
                    continue
                with open(candidate, "r", encoding="utf-8") as f:
                    cached_plan = json.load(f)
                cached_n = len(cached_plan.get("features", []) or [])
                if cached_n != requested_n:
                    print(
                        f"  [reproduce] Ignoring cached feature_plan with {cached_n} features "
                        f"(requested {requested_n}): {candidate}"
                    )
                    continue
                feature_plan = cached_plan
                print(
                    f"  [reproduce] Reusing cached feature_plan "
                    f"({cached_n} features): {candidate}"
                )
                break

        if feature_plan is None:
            final_state = app.invoke(initial_state)
            feature_plan = final_state.get("feature_plan")
        
        if not feature_plan or "features" not in feature_plan:
            print("❌ Error: failed to generate a feature plan")
            if round_num == 1:
                sys.exit(1)
            else:
                print("  ⚠️  Skipping this round, continuing to the next")
                continue

        # If the user query explicitly lists feature names (1) xxx_feature: ...), force the use of that list to keep the planner on track
        explicit_feature_specs = _extract_explicit_feature_specs(args.user_query)
        if explicit_feature_specs:
            planned_features = feature_plan.get("features", []) or []
            planned_by_norm = {
                _normalize_feature_key(f.get("name", "")): f
                for f in planned_features
                if isinstance(f, dict) and f.get("name")
            }
            enforced_features: List[Dict[str, Any]] = []
            for spec in explicit_feature_specs:
                feature_name = spec["name"]
                normalized_name = _normalize_feature_key(feature_name)
                planned = planned_by_norm.get(normalized_name, {})
                feature_entry = {
                    "name": feature_name,
                    "description": spec.get("description") or planned.get("description") or f"Explicitly requested feature: {feature_name}.",
                    "category": planned.get("category", "other"),
                    "needs_segmentation": planned.get("needs_segmentation", True),
                    "segmentation_prompt": planned.get(
                        "segmentation_prompt",
                        "Segment mitochondria-related structures and cell region as needed for this feature."
                    ),
                    "method": planned.get("method", "code"),
                    "method_rationale": planned.get(
                        "method_rationale",
                        "This feature is explicitly specified by the user and should be computed deterministically."
                    ),
                }
                enforced_features.append(feature_entry)

            feature_plan["features"] = enforced_features
            print(
                f"  🔒 Detected an explicit user feature list; forcing the use of {len(enforced_features)} specified features (in query order)."
            )
        
        print(f"  Generated {len(feature_plan['features'])} features")
        
        # Save the feature plan
        import json
        plan_path = round_results_dir / "feature_plan.json"
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(feature_plan, f, indent=2, ensure_ascii=False)
        print(f"  Feature plan saved: {plan_path}")

        if (
            settings.reproduce_mode
            and settings.reproduce_cache_dir
            and round_num == 1
            and plan_cache_path is not None
            and not plan_cache_path.exists()
        ):
            plan_cache_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(plan_path, plan_cache_path)
            print(f"  [reproduce] Wrote canonical feature_plan cache: {plan_cache_path}")
    
        # Step 3.5: Data preprocessing - generate slices for VLM features (if they do not exist)
        # Only run preprocessing in the first round
        if round_num == 1:
            print(f"\nStep 3.5: Data preprocessing - check and generate the slices directory")
            from utils_modules.data_preprocessing import preprocess_all_samples
            
            # Read the content of the description file (if it exists)
            description_text = None
            if description_path and description_path.exists():
                try:
                    with open(description_path, 'r', encoding='utf-8') as f:
                        description_text = f.read()
                except Exception as e:
                    print(f"  ⚠️  Warning: failed to read the description file: {e}")
            
            # Preprocess all samples to ensure the slices directory exists
            preprocess_results = preprocess_all_samples(
                data_root,
                sample_ids,
                description_text,
                secondary_dir_name="slices"
            )
        
        # Step 4: Batch feature extraction (process all samples feature by feature)
        print(f"\nStep 4: Batch feature extraction (process all samples feature by feature)")
        features = feature_plan.get("features", [])
        
        # According to the method restriction: for code/vlm, keep all features and compute them uniformly with that method, and normalize feature names (code strips the vlm_ prefix, vlm adds the vlm_ prefix)
        if args.method == "code":
            features = [{**f, "method": "code", "name": _feature_name_for_method(f.get("name", ""), "code")} for f in features]
            print(f"  📌 Method restricted to code: all {len(features)} features are computed using code (the vlm_ prefix has been removed from names)")
        elif args.method == "vlm":
            features = [{**f, "method": "vlm", "name": _feature_name_for_method(f.get("name", ""), "vlm")} for f in features]
            print(f"  📌 Method restricted to vlm: all {len(features)} features are computed using VLM (the vlm_ prefix has been added to names)")
        
        # Allocate features according to code-vlm-ratio (only effective when method=="both")
        if args.method == "both" and len(features) > 0:
            # Validate the ratio range
            code_ratio = max(0.0, min(1.0, args.code_vlm_ratio))
            vlm_ratio = 1.0 - code_ratio
            
            # Separate code and vlm features
            code_features_list = [f for f in features if f.get("method", "code") == "code"]
            vlm_features_list = [f for f in features if f.get("method", "code") == "vlm"]
            
            total_features = len(features)
            target_code_count = int(total_features * code_ratio)
            target_vlm_count = total_features - target_code_count
            
            # If the current allocation does not match the ratio, adjust it
            current_code_count = len(code_features_list)
            current_vlm_count = len(vlm_features_list)
            
            if current_code_count + current_vlm_count > 0:
                # Compute how many need to be adjusted
                code_diff = target_code_count - current_code_count
                vlm_diff = target_vlm_count - current_vlm_count
                
                # If the difference is large, reallocate
                if abs(code_diff) > 1 or abs(vlm_diff) > 1:
                    print(f"  📊 Adjusting feature allocation according to code-vlm-ratio ({code_ratio:.1%} code, {vlm_ratio:.1%} vlm)...")
                    print(f"     Current: {current_code_count} code, {current_vlm_count} vlm")
                    print(f"     Target: {target_code_count} code, {target_vlm_count} vlm")
                    
                    # If there are too many code features, convert some to vlm
                    if code_diff < 0:
                        to_convert = min(abs(code_diff), len(code_features_list))
                        for i in range(to_convert):
                            code_features_list[i]["method"] = "vlm"
                            vlm_features_list.append(code_features_list[i])
                        code_features_list = code_features_list[to_convert:]
                    # If there are too many vlm features, convert some to code
                    elif vlm_diff < 0:
                        to_convert = min(abs(vlm_diff), len(vlm_features_list))
                        for i in range(to_convert):
                            vlm_features_list[i]["method"] = "code"
                            code_features_list.append(vlm_features_list[i])
                        vlm_features_list = vlm_features_list[to_convert:]
                    
                    # Recombine
                    features = code_features_list + vlm_features_list
                    print(f"     After adjustment: {len(code_features_list)} code, {len(vlm_features_list)} vlm")
                else:
                    print(f"  📊 Feature allocation meets the ratio requirement: {current_code_count} code ({current_code_count/total_features:.1%}), {current_vlm_count} vlm ({current_vlm_count/total_features:.1%})")
        
        if not features:
            print("  ⚠️  No features to extract")
            all_results: Dict[str, Dict[str, Any]] = {}
            # If the CSV file does not exist, create an empty CSV file (with only the sample_id column)
            if not features_csv_path.exists():
                empty_df = pd.DataFrame({'sample_id': sample_ids})
                empty_df.to_csv(features_csv_path, index=False, encoding='utf-8')
        else:
            print(f"  {len(features)} features total, {len(sample_ids)} samples")
            
            # Initialize the results dictionary: one dict per sample
            all_results: Dict[str, Dict[str, Any]] = {sample_id: {} for sample_id in sample_ids}
            
            # Read or initialize the CSV file
            if features_csv_path.exists() and round_num > 1:
                # If the file already exists (not the first round), read the existing data
                features_df = pd.read_csv(features_csv_path)
                print(f"  ✅ Read existing CSV file: {features_csv_path} ({len(features_df.columns) - 1} features already present)")
            else:
                # Initialize the CSV file: create a DataFrame with only the sample_id column
                features_df = pd.DataFrame({'sample_id': sample_ids})
                features_df.to_csv(features_csv_path, index=False, encoding='utf-8')
                print(f"  ✅ Initialized CSV file: {features_csv_path}")
            
            # Separate VLM features and code features
            vlm_features = [f for f in features if f.get("method", "code") == "vlm"]
            code_features = [f for f in features if f.get("method", "code") == "code"]
            
            # Batch-process VLM features: process all VLM features at once for each sample
            if vlm_features:
                print(f"\n  Batch-processing {len(vlm_features)} VLM features (all VLM features processed at once per sample)")
                
                # The online VLM does not need a local GPU, and multiprocessing spawn loses the apply_vlm_provider state, so force a single process
                if args.multigpu and settings.vlm_api_provider == "online":
                    print("  ⚠️  When vlm-api-provider=online, multi-GPU is not used; forcing single-process online API calls")
                    args.multigpu = False

                # Check whether multi-GPU is enabled
                if args.multigpu:
                    # Multi-GPU mode: parallel processing
                    available_gpus = _detect_available_gpus()
                    if not available_gpus:
                        print("  ⚠️  Warning: no available GPU detected, falling back to single-GPU mode")
                        use_multigpu = False
                    else:
                        use_multigpu = True
                        num_gpus = len(available_gpus)
                        print(f"  ✅ Detected {num_gpus} GPUs: {available_gpus}")
                else:
                    use_multigpu = False
                    available_gpus = []
                
                # Online API concurrency mode: VLM scoring is I/O bound (network round trips), so use a thread pool to process multiple samples concurrently
                online_concurrent = (
                    settings.vlm_api_provider == "online"
                    and int(getattr(args, "vlm_online_concurrency", 1)) > 1
                )
                if online_concurrent:
                    from nodes.execution import _execute_vlm_features_batch
                    from state import AgentState
                    from utils_helpers import select_appropriate_data_source
                    from tools.segmentation import check_segmentation_exists
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    import threading
                    import time as _time

                    concurrency = max(1, int(getattr(args, "vlm_online_concurrency", 8)))
                    needs_segmentation = (any(f.get("needs_segmentation", False) for f in vlm_features) and args.enable_segmentation)

                    batch_log_file = None
                    if round_results_dir:
                        batch_log_dir = round_results_dir / "features" / "vlm_batch"
                        batch_log_dir.mkdir(parents=True, exist_ok=True)
                        batch_log_file = batch_log_dir / "execution_log.txt"
                    log_lock = threading.Lock()
                    results_lock = threading.Lock()

                    def _process_one_vlm_sample(sample_id):
                        sample_dir = data_root / sample_id
                        if not sample_dir.exists():
                            return sample_id, None, "Sample directory does not exist"
                        image_paths = select_appropriate_data_source(
                            sample_dir, vlm_features[0], dataset_description
                        )
                        if not image_paths:
                            return sample_id, None, "No image files found"
                        sample_seg_mask = None
                        if needs_segmentation:
                            existing_seg = check_segmentation_exists(sample_dir)
                            if existing_seg:
                                sample_seg_mask = str(existing_seg)
                        state: AgentState = {
                            "messages": [], "user_query": "", "sample_id": sample_id,
                            "image_paths": image_paths, "research_summary": dataset_description or "",
                            "expert_examples": [], "expert_knowledge": expert_knowledge,
                            "deep_research": deep_research, "rag_knowledge": rag_knowledge,
                            "feature_plan": feature_plan,
                            "segmentation_mask": sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                            "analysis_results": {}, "current_step": "execution",
                            "iteration_count": 0, "error_log": [],
                            "features_list": [{"name": f.get("name", ""), "description": f.get("description", ""), "category": f.get("category", "")} for f in vlm_features],
                            "num_features": len(vlm_features),
                        }
                        max_attempts = 3
                        for attempt in range(1, max_attempts + 1):
                            try:
                                batch_results = _execute_vlm_features_batch(
                                    vlm_features, image_paths, state,
                                    segmentation_mask=sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                                    log_file=str(batch_log_file) if batch_log_file else None,
                                )
                                if batch_results and any(v is not None for v in batch_results.values()):
                                    return sample_id, batch_results, None
                                if batch_results:
                                    return sample_id, batch_results, "All results are None"
                                return sample_id, None, "Empty result"
                            except Exception as e:
                                if attempt >= max_attempts:
                                    return sample_id, None, f"{type(e).__name__}: {e}"
                                _time.sleep(1)
                        return sample_id, None, "Unknown error"

                    print(f"  🚀 Online API concurrency mode: {concurrency} concurrent threads, processing {len(sample_ids)} samples")
                    completed = 0
                    with tqdm(total=len(sample_ids), desc="  VLM batch scoring (concurrent)", leave=False, ncols=80) as pbar:
                        with ThreadPoolExecutor(max_workers=concurrency) as pool:
                            futures = {pool.submit(_process_one_vlm_sample, sid): sid for sid in sample_ids}
                            for fut in as_completed(futures):
                                sample_id, batch_results, err = fut.result()
                                with results_lock:
                                    if batch_results:
                                        for feat_name, value in batch_results.items():
                                            if sample_id in all_results:
                                                all_results[sample_id][feat_name] = value
                                            else:
                                                all_results[sample_id] = {feat_name: value}
                                    elif sample_id not in all_results:
                                        all_results[sample_id] = {}
                                if err and batch_log_file:
                                    with log_lock:
                                        with open(batch_log_file, 'a', encoding='utf-8') as _lf:
                                            _lf.write(f"{sample_id}: ⚠️  {err}\n")
                                completed += 1
                                pbar.update(1)
                    print(f"  ✅ Online concurrent processing complete: {completed}/{len(sample_ids)} samples")

                elif use_multigpu:
                    # Multi-GPU parallel processing
                    from nodes.execution import _execute_vlm_features_batch
                    from state import AgentState
                    from utils_helpers import select_appropriate_data_source
                    from tools.segmentation import check_segmentation_exists
                    from tools.segmentation_tool import get_segmentation_tool
                    
                    # Only segment when data segmentation is enabled and the feature requires it; otherwise the seg the VLM receives is None
                    needs_segmentation = (any(f.get("needs_segmentation", False) for f in vlm_features) and args.enable_segmentation)
                    
                    # Create the task queue and result queue
                    # Note: in spawn mode the queues must be created in the main process and then passed to the child processes
                    mp_context = mp.get_context('spawn')  # create the context up front to ensure the queues use the same context
                    task_queue = mp_context.Queue()
                    result_queue = mp_context.Queue()
                    
                    # Number the samples in order and distribute them to the queue (preserving order)
                    for idx, sample_id in enumerate(sample_ids):
                        task_queue.put((idx, sample_id))
                    
                    # Add end signals (one per GPU)
                    for gpu_id in available_gpus:
                        task_queue.put(None)
                    
                    # Start the worker processes
                    processes = []
                    for gpu_id in available_gpus:
                        # Note: spawn requires serializable arguments, so Path objects need to be converted to strings
                        p = mp_context.Process(
                            target=_vlm_worker_process,
                            args=(
                                gpu_id,
                                task_queue,
                                result_queue,
                                vlm_features,
                                str(data_root),  # convert to string, then convert back to Path in the worker process
                                feature_plan,
                                dataset_description,
                                expert_knowledge,
                                deep_research,
                                rag_knowledge,
                                str(round_results_dir) if round_results_dir else None,  # convert to string
                                needs_segmentation
                            )
                        )
                        p.start()
                        processes.append(p)
                    
                    # Wait a short while to let the processes start
                    import time
                    time.sleep(0.5)
                    
                    # Collect results (sorted by sample_index, preserving order)
                    results_dict = {}  # key: sample_index, value: (sample_id, results)
                    completed_count = 0
                    total_samples = len(sample_ids)
                    
                    print(f"  🚀 Started {len(processes)} worker processes, processing {total_samples} samples in parallel")
                    
                    # Use a progress bar to show progress
                    with tqdm(total=total_samples, desc="  VLM batch scoring (multi-GPU)", leave=False, ncols=80) as pbar:
                        while completed_count < total_samples:
                            try:
                                sample_index, sample_id, batch_results = result_queue.get(timeout=60)
                                results_dict[sample_index] = (sample_id, batch_results)
                                completed_count += 1
                                pbar.update(1)
                            except queue.Empty:
                                # Queue timed out, check whether the processes are still running
                                alive_processes = [p.is_alive() for p in processes]
                                if all(not p.is_alive() for p in processes):
                                    # All processes have finished, but there may still be results in the queue
                                    while not result_queue.empty():
                                        try:
                                            sample_index, sample_id, batch_results = result_queue.get_nowait()
                                            results_dict[sample_index] = (sample_id, batch_results)
                                            completed_count += 1
                                            pbar.update(1)
                                        except queue.Empty:
                                            break
                                    break
                            except Exception as e:
                                # Other exceptions: log and continue
                                print(f"  ⚠️  Error occurred while collecting results: {e}")
                                # Check whether the processes are still running
                                if all(not p.is_alive() for p in processes):
                                    break
                    
                    # Wait for all processes to finish
                    for p in processes:
                        p.join(timeout=10)
                        if p.is_alive():
                            p.terminate()
                            p.join()
                    
                    # Organize results in sample_index order and add them to all_results (preserving the original sample_ids order)
                    for sample_index in sorted(results_dict.keys()):
                        sample_id, batch_results = results_dict[sample_index]
                        if batch_results:
                            for feat_name, value in batch_results.items():
                                if sample_id in all_results:
                                    all_results[sample_id][feat_name] = value
                                else:
                                    all_results[sample_id] = {feat_name: value}
                        else:
                            # Even if batch_results is empty, ensure sample_id is in all_results
                            if sample_id not in all_results:
                                all_results[sample_id] = {}
                    
                    print(f"  ✅ Multi-GPU processing complete: {completed_count}/{total_samples} samples")
                    
                else:
                    # Single-GPU mode: keep the original logic
                    from nodes.execution import _execute_vlm_features_batch
                    from state import AgentState
                    from utils_helpers import select_appropriate_data_source
                    from tools.segmentation import check_segmentation_exists
                    from tools.segmentation_tool import get_segmentation_tool
                    
                    # Create the log file for batch VLM features
                    batch_log_file = None
                    if round_results_dir:
                        batch_log_dir = round_results_dir / "features" / "vlm_batch"
                        batch_log_dir.mkdir(parents=True, exist_ok=True)
                        batch_log_file = batch_log_dir / "execution_log.txt"
                        batch_log_fp = open(batch_log_file, 'w', encoding='utf-8')
                    else:
                        batch_log_fp = None
                    
                    # Batch-process all VLM features for each sample
                    try:
                        for sample_id in tqdm(sample_ids, desc="  VLM batch scoring", leave=False, ncols=80):
                            if batch_log_fp:
                                batch_log_fp.write(f"\n{'='*60}\n")
                                batch_log_fp.write(f"Processing sample: {sample_id}\n")
                                batch_log_fp.write(f"Features: {[f.get('name') for f in vlm_features]}\n")
                                batch_log_fp.write(f"{'='*60}\n")
                                batch_log_fp.flush()
                            
                            sample_dir = data_root / sample_id
                            if not sample_dir.exists():
                                error_msg = f"Sample directory does not exist: {sample_dir}"
                                if batch_log_fp:
                                    batch_log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                                continue
                            
                            # Intelligently select the data source (using the requirements of the first VLM feature)
                            image_paths = select_appropriate_data_source(
                                sample_dir, 
                                vlm_features[0], 
                                dataset_description
                            )
                            if not image_paths:
                                error_msg = f"No image files found for sample {sample_id}"
                                if batch_log_fp:
                                    batch_log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                                continue
                            
                            # Check and run segmentation (only when data segmentation is enabled and the feature requires it)
                            sample_seg_mask = None
                            needs_segmentation = (any(f.get("needs_segmentation", False) for f in vlm_features) and args.enable_segmentation)
                            if needs_segmentation:
                                existing_seg = check_segmentation_exists(sample_dir)
                                if existing_seg:
                                    sample_seg_mask = str(existing_seg)
                                else:
                                    seg_tool = get_segmentation_tool()
                                    channels = vlm_features[0].get("segmentation_channels")
                                    seg_result = seg_tool.invoke({
                                        "sample_dir": str(sample_dir),
                                        "image_path": image_paths[0],
                                        "channels": channels,
                                        "conda_env": settings.segmentation_conda_env
                                    })
                                    if seg_result.get("success"):
                                        sample_seg_mask = seg_result.get("mask_path")
                                    else:
                                        if batch_log_fp:
                                            batch_log_fp.write(f"{sample_id}: ❌ Segmentation failed\n")
                                        continue
                            
                            # Build AgentState for batch VLM execution
                            state: AgentState = {
                                "messages": [],
                                "user_query": "",
                                "sample_id": sample_id,
                                "image_paths": image_paths,
                                "research_summary": dataset_description or "",
                                "expert_examples": [],
                                "expert_knowledge": expert_knowledge,
                                "deep_research": deep_research,
                                "rag_knowledge": rag_knowledge,
                                "feature_plan": feature_plan,
                                "segmentation_mask": sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                                "analysis_results": {},
                                "current_step": "execution",
                                "iteration_count": 0,
                                "error_log": [],
                                # Batch feature information
                                "features_list": [{"name": f.get("name", ""), "description": f.get("description", ""), "category": f.get("category", "")} for f in vlm_features],
                                "num_features": len(vlm_features),
                            }
                            
                            # Run batch VLM features (single sample, one call processes all VLM features)
                            max_attempts = 3
                            attempt_count = 0
                            batch_results = None
                            success = False
                            
                            while attempt_count < max_attempts and not success:
                                attempt_count += 1
                                try:
                                    if attempt_count > 1:
                                        retry_msg = f"{sample_id}: 🔄 Retry #{attempt_count - 1} (of {max_attempts} attempts)"
                                        print(f"    {retry_msg}")
                                        if batch_log_fp:
                                            batch_log_fp.write(f"{retry_msg}\n")
                                            batch_log_fp.flush()
                                    
                                    batch_results = _execute_vlm_features_batch(
                                        vlm_features,
                                        image_paths,
                                        state,
                                        segmentation_mask=sample_seg_mask if (needs_segmentation and sample_seg_mask is not None) else None,
                                        log_file=str(batch_log_file) if batch_log_file else None
                                    )
                                    
                                    if batch_results:
                                        # Check whether there are any valid results
                                        has_valid_results = any(v is not None for v in batch_results.values())
                                        if has_valid_results:
                                            success = True
                                        # Add the batch results to all_results (add even if some are None)
                                        for feat_name, value in batch_results.items():
                                            if sample_id in all_results:
                                                all_results[sample_id][feat_name] = value
                                            else:
                                                all_results[sample_id] = {feat_name: value}
                                        
                                        if batch_log_fp:
                                            if has_valid_results:
                                                batch_log_fp.write(f"{sample_id}: ✅ Batch scoring complete\n")
                                            else:
                                                batch_log_fp.write(f"{sample_id}: ⚠️  Batch scoring complete but all results are None\n")
                                            for feat_name, value in batch_results.items():
                                                batch_log_fp.write(f"  {feat_name}: {value}\n")
                                            batch_log_fp.flush()
                                    else:
                                        if batch_log_fp:
                                            batch_log_fp.write(f"{sample_id}: ⚠️  Batch feature extraction returned empty results\n")
                                        break
                                except TimeoutError as e:
                                    error_msg = f"{sample_id}: ❌ Timeout error (attempt {attempt_count}/{max_attempts}): {e}"
                                    if batch_log_fp:
                                        batch_log_fp.write(f"{error_msg}\n")
                                        batch_log_fp.flush()
                                    
                                    if attempt_count >= max_attempts:
                                        final_error = f"{sample_id}: ❌ Timeout error, already tried {max_attempts} times, skipping this sample"
                                        print(f"    {final_error}")
                                        if batch_log_fp:
                                            batch_log_fp.write(f"{final_error}\n")
                                            batch_log_fp.flush()
                                    else:
                                        import time
                                        time.sleep(1)
                                except Exception as e:
                                    error_msg = f"{sample_id}: ❌ {e}"
                                    if batch_log_fp:
                                        batch_log_fp.write(f"{error_msg}\n")
                                    import traceback
                                    if batch_log_fp:
                                        batch_log_fp.write(traceback.format_exc() + "\n")
                                    break
                    finally:
                        if batch_log_fp:
                            batch_log_fp.close()
                    
                    # Log output for single-GPU mode
                    if batch_log_file:
                        print(f"    Detailed batch VLM results saved to: {batch_log_file}")
                
                # Batch-update the CSV file: add all VLM feature columns (shared by single-GPU and multi-GPU modes)
                # Note: this code is outside both the if use_multigpu and else blocks, ensuring both modes execute it
                if vlm_features:  # ensure there are VLM features to update
                    try:
                        features_df = pd.read_csv(features_csv_path)
                        if 'sample_id' not in features_df.columns:
                            features_df['sample_id'] = sample_ids
                        else:
                            # Ensure it is reordered according to the order of sample_ids
                            features_df = features_df.set_index('sample_id').reindex(sample_ids).reset_index()
                        
                        for feature in vlm_features:
                            original_feature_name = feature.get('name', 'unknown')
                            feature_name = original_feature_name
                            
                            # Check whether the feature already exists
                            if feature_name in features_df.columns:
                                existing_values = features_df[feature_name]
                                valid_count = existing_values.notna().sum()
                                if valid_count == 0:
                                    print(f"  ⚠️  Feature '{feature_name}' already exists but is all NaN, will be updated")
                                else:
                                    # Even if the feature already exists, generate a new feature (by adding a suffix)
                                    print(f"  ℹ️  Feature '{feature_name}' already exists ({valid_count}/{len(existing_values)} valid values), will generate a new version")
                                    # Add a timestamp suffix to ensure the feature name is unique
                                    timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
                                    feature_name = f"{feature_name}_new{timestamp_suffix}"
                                    print(f"  → New feature name: {feature_name}")
                                    # Update the feature name in all_results (from original_feature_name to feature_name)
                                    for sample_id in sample_ids:
                                        if sample_id in all_results and original_feature_name in all_results[sample_id]:
                                            all_results[sample_id][feature_name] = all_results[sample_id].pop(original_feature_name)
                            
                            # Extract feature values in the order of sample_ids (to ensure correct order)
                            feature_values = []
                            for sample_id in sample_ids:
                                # Prefer the new feature name; if it does not exist, use the original feature name
                                value = all_results.get(sample_id, {}).get(feature_name)
                                if value is None and feature_name != original_feature_name:
                                    # If the new feature name does not exist, try the original feature name
                                    value = all_results.get(sample_id, {}).get(original_feature_name)
                                if value is None:
                                    value = np.nan
                                feature_values.append(value)
                            
                            features_df[feature_name] = feature_values
                            valid_count = sum(1 for v in feature_values if not pd.isna(v))
                            print(f"  ✅ Updated CSV: {feature_name} ({valid_count}/{len(feature_values)} valid values)")
                        
                        # Ensure sample_id is the first column, and sort by the sample_ids order
                        cols = ['sample_id'] + [col for col in features_df.columns if col != 'sample_id']
                        features_df = features_df[cols]
                        features_df.to_csv(features_csv_path, index=False, encoding='utf-8')
                        print(f"  ✅ CSV file saved: {features_csv_path}")
                    except Exception as e:
                        print(f"  ⚠️  Batch CSV update failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Optimized code feature processing logic: first generate and test all code, then merge and execute
            if code_features:
                print(f"\n  Processing {len(code_features)} code features (optimized mode: test first, then merge and execute)")
                
                # Step 1: Generate and test code for each feature (do not run all samples)
                from tools.code_executor import run_code_generation_test_only
                from state import AgentState
                
                successful_features = []
                successful_code_paths = []
                failed_features = []
                
                for i, feature in enumerate(code_features, 1):
                    original_feature_name = feature.get('name', f'feature_{i}')
                    print(f"\n  Feature [{i}/{len(code_features)}]: {original_feature_name}")
                    print(f"    Method: code (generate and test)")
                    
                    # Build AgentState
                    state: AgentState = {
                        "messages": [],
                        "user_query": "",
                        "sample_id": sample_ids[0] if sample_ids else "",
                        "image_paths": [],
                        "research_summary": dataset_description or "",
                        "expert_examples": [],
                        "expert_knowledge": expert_knowledge,
                        "deep_research": deep_research,
                        "rag_knowledge": rag_knowledge,
                        "feature_plan": feature_plan,
                        "segmentation_mask": None,
                        "analysis_results": {},
                        "current_step": "execution",
                        "iteration_count": 0,
                        "error_log": [],
                        "feature_name": feature.get("name", ""),
                        "feature_description": feature.get("description", ""),
                        "feature_category": feature.get("category", ""),
                    }
                    
                    # Try to read mask order information from the segmentation summary
                    segmentation_mask_order = ""
                    try:
                        if round_results_dir:
                            segmentation_summary_path = round_results_dir / "segmentation_summary.json"
                            if segmentation_summary_path.exists():
                                import json
                                with open(segmentation_summary_path, 'r', encoding='utf-8') as f:
                                    summary = json.load(f)
                                    segmentation_mask_order = summary.get("mask_order_description", "")
                                    state["segmentation_mask_order"] = segmentation_mask_order
                    except Exception:
                        pass
                    
                    # Get the data path selector
                    from tools.data_path_selector import get_data_path_selector
                    selector = get_data_path_selector(verbose=False)
                    
                    def find_code_data_sources(sample_dir, _):
                        temp_feature = {"method": "code"}
                        result = selector.select_data_paths(
                            Path(sample_dir),
                            temp_feature,
                            dataset_description,
                            method="code"
                        )
                        if isinstance(result, dict):
                            return result.get("image_paths", [])
                        else:
                            return result if isinstance(result, list) else []
                    
                    # Generate and test the code (test only the first sample)
                    extract_py_path, code_result = run_code_generation_test_only(
                        feature,
                        state,
                        sample_ids,
                        data_root,
                        find_code_data_sources,
                        round_results_dir,
                        conda_env=None,
                        max_cycles=None,
                        segmentation_mask_path=None,
                        enable_critic=settings.enable_critic_agent  # pass the critic agent enabled state
                    )
                    
                    if extract_py_path and extract_py_path.exists():
                        print(f"    ✅ Code generation and test succeeded: {extract_py_path}")
                        successful_features.append(feature)
                        successful_code_paths.append(extract_py_path)
                    else:
                        print(f"    ❌ Code generation or test failed, skipping this feature")
                        failed_features.append(feature)
                
                # Step 2: If there are successful features, merge the code and execute
                if successful_features:
                    print(f"\n  Step 2: Merging the code of {len(successful_features)} features...")
                    
                    from tools.code_executor import merge_feature_codes, execute_merged_code
                    
                    # Merge the code
                    merged_code, merge_response = merge_feature_codes(
                        successful_features,
                        successful_code_paths,
                        state,
                        round_results_dir
                    )
                    
                    if merged_code:
                        print(f"  ✅ Code merged successfully")
                        
                        # Step 3: Execute the merged code
                        print(f"\n  Step 3: Executing the merged code, processing all samples...")
                        
                        feature_names = [f.get("name", f"feature_{i}") for i, f in enumerate(successful_features, 1)]
                        
                        merged_result = execute_merged_code(
                            merged_code,
                            feature_names,
                            sample_ids,
                            data_root,
                            find_code_data_sources,
                            round_results_dir,
                            conda_env=None,
                            segmentation_mask_path=None,
                            num_workers=args.code_parallel_workers
                        )
                        
                        # Step 4: Organize the results and save to CSV
                        print(f"\n  Step 4: Organizing feature results and saving to CSV...")
                        
                        # Convert the result format: merged_result.values is in the {sample_id: {feature_name: value}} format
                        # It needs to be converted to one column per feature
                        try:
                            features_df = pd.read_csv(features_csv_path)
                            if 'sample_id' not in features_df.columns:
                                features_df['sample_id'] = sample_ids
                            else:
                                features_df = features_df.set_index('sample_id').reindex(sample_ids).reset_index()
                            
                            # Add each feature column in order
                            for feature_name in feature_names:
                                original_feature_name = feature_name
                                
                                # Check whether the feature already exists
                                if feature_name in features_df.columns:
                                    existing_values = features_df[feature_name]
                                    valid_count = existing_values.notna().sum()
                                    if valid_count == 0:
                                        print(f"  ⚠️  Feature '{feature_name}' already exists but is all NaN, will be updated")
                                    else:
                                        # Generate a new version
                                        print(f"  ℹ️  Feature '{feature_name}' already exists ({valid_count}/{len(existing_values)} valid values), will generate a new version")
                                        timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
                                        feature_name = f"{feature_name}_new{timestamp_suffix}"
                                        print(f"  → New feature name: {feature_name}")
                                
                                # Extract feature values (in sample_ids order); the merged code returns keys that are mostly snake_case, so resolve using the display name
                                feature_values = []
                                for sample_id in sample_ids:
                                    if sample_id in merged_result.values:
                                        sample_features = merged_result.values[sample_id]
                                        value = _get_merged_feature_value(sample_features, original_feature_name)
                                        if value is None:
                                            value = np.nan
                                    elif sample_id in merged_result.errors:
                                        value = np.nan
                                    else:
                                        value = np.nan
                                    feature_values.append(value)
                                
                                # Add to the DataFrame
                                features_df[feature_name] = feature_values
                                
                                # Update all_results
                                for sample_id, value in zip(sample_ids, feature_values):
                                    if sample_id in all_results:
                                        all_results[sample_id][feature_name] = value
                                    else:
                                        all_results[sample_id] = {feature_name: value}
                                
                                valid_count = sum(1 for v in feature_values if not pd.isna(v))
                                print(f"  ✅ Added feature column: {feature_name} ({valid_count}/{len(feature_values)} valid values)")
                            
                            # Ensure sample_id is the first column, and sort by the sample_ids order
                            cols = ['sample_id'] + [col for col in features_df.columns if col != 'sample_id']
                            features_df = features_df[cols]
                            features_df.to_csv(features_csv_path, index=False, encoding='utf-8')
                            print(f"  ✅ CSV file saved: {features_csv_path}")
                            
                        except Exception as e:
                            print(f"  ⚠️  Failed to organize results and save CSV: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Handle the failed samples
                        if merged_result.errors:
                            print(f"  ⚠️  {len(merged_result.errors)} samples failed to execute")
                            for sample_id, error_msg in list(merged_result.errors.items())[:5]:
                                print(f"    {sample_id}: {error_msg[:200]}")
                    else:
                        print(f"  ❌ Code merge failed, falling back to per-feature execution mode")
                        # Fallback: run each feature separately on all samples, then aggregate the results and write the CSV
                        from tools.code_executor import CodeExecutor, ExtractionResult
                        executor = CodeExecutor(data_root)
                        feature_names = [f.get("name", f"feature_{i}") for i, f in enumerate(successful_features, 1)]
                        fallback_values = {sid: {} for sid in sample_ids}
                        fallback_errors = {}
                        for i, (feature, extract_py_path) in enumerate(zip(successful_features, successful_code_paths), 1):
                            fn = feature.get("name", "unknown")
                            print(f"\n  Fallback mode: running feature {i}/{len(successful_features)}: {fn}")
                            log_file = None
                            if round_results_dir:
                                log_dir = round_results_dir / "features" / fn
                                log_dir.mkdir(parents=True, exist_ok=True)
                                log_file = log_dir / "execution_log.txt"
                            try:
                                extraction_result = executor.execute_all_samples(
                                    extract_py_path,
                                    sample_ids,
                                    find_code_data_sources,
                                    log_file=log_file,
                                    segmentation_mask_path=None,
                                )
                                for sid, val in extraction_result.values.items():
                                    fallback_values[sid][fn] = val
                                for sid in sample_ids:
                                    if sid not in extraction_result.values:
                                        fallback_values[sid][fn] = np.nan
                                for sid, err in extraction_result.errors.items():
                                    fallback_errors[sid] = fallback_errors.get(sid, "") + f"[{fn}] {err}; "
                            except Exception as e:
                                print(f"    ⚠️  Feature {fn} raised an exception during execution: {e}")
                                for sid in sample_ids:
                                    fallback_values[sid][fn] = np.nan
                        # Reuse Step 4: write the CSV and update all_results with the fallback results
                        merged_result = ExtractionResult(values=fallback_values, errors=fallback_errors)
                        print(f"\n  Step 4: Organizing feature results and saving to CSV (fallback mode)...")
                        try:
                            features_df = pd.read_csv(features_csv_path)
                            if "sample_id" not in features_df.columns:
                                features_df["sample_id"] = sample_ids
                            else:
                                features_df = features_df.set_index("sample_id").reindex(sample_ids).reset_index()
                            for feature_name in feature_names:
                                original_feature_name = feature_name
                                if feature_name in features_df.columns:
                                    existing_values = features_df[feature_name]
                                    valid_count = existing_values.notna().sum()
                                    if valid_count == 0:
                                        print(f"  ⚠️  Feature '{feature_name}' already exists but is all NaN, will be updated")
                                    else:
                                        timestamp_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
                                        feature_name = f"{feature_name}_new{timestamp_suffix}"
                                        print(f"  → New feature name: {feature_name}")
                                feature_values = []
                                for sample_id in sample_ids:
                                    if sample_id in merged_result.values:
                                        sample_features = merged_result.values[sample_id]
                                        value = _get_merged_feature_value(sample_features, original_feature_name)
                                        if value is None:
                                            value = np.nan
                                    elif sample_id in merged_result.errors:
                                        value = np.nan
                                    else:
                                        value = np.nan
                                    feature_values.append(value)
                                features_df[feature_name] = feature_values
                                for sample_id, value in zip(sample_ids, feature_values):
                                    if sample_id in all_results:
                                        all_results[sample_id][feature_name] = value
                                    else:
                                        all_results[sample_id] = {feature_name: value}
                                valid_count = sum(1 for v in feature_values if not pd.isna(v))
                                print(f"  ✅ Added feature column: {feature_name} ({valid_count}/{len(feature_values)} valid values)")
                            cols = ["sample_id"] + [c for c in features_df.columns if c != "sample_id"]
                            features_df = features_df[cols]
                            features_df.to_csv(features_csv_path, index=False, encoding="utf-8")
                            print(f"  ✅ CSV file saved: {features_csv_path}")
                        except Exception as e:
                            print(f"  ⚠️  Failed to organize results and save CSV: {e}")
                            import traceback
                            traceback.print_exc()
                        if merged_result.errors:
                            print(f"  ⚠️  {len(merged_result.errors)} samples had execution failures")
                            for sample_id, error_msg in list(merged_result.errors.items())[:5]:
                                print(f"    {sample_id}: {error_msg[:200]}")
                else:
                    print(f"  ⚠️  No features had successfully generated code, skipping code feature processing")
                
                # Report the failed features
                if failed_features:
                    print(f"\n  ⚠️  {len(failed_features)} features failed code generation or testing:")
                    for feature in failed_features:
                        print(f"    - {feature.get('name', 'unknown')}")
        
        # Step 5: Dataset-level analysis
        print(f"\nStep 5: Dataset-level analysis (round {round_num})")
        round_validation_payload = None
        if all_results:
            analysis_result = process_dataset_level_analysis(all_results, data_root)
            
            # Validate the CSV file (features were already saved in real time in step 4)
            print(f"\nStep 5.5: Validating the feature CSV file...")
            try:
                if features_csv_path.exists():
                    features_df = pd.read_csv(features_csv_path)
                    print(f"  ✅ Feature CSV file exists: {features_csv_path} (shape: {features_df.shape})")
                    print(f"     Contains {len(features_df.columns) - 1} feature columns")
                else:
                    print(f"  ⚠️  Warning: feature CSV file does not exist: {features_csv_path}")
            except Exception as e:
                print(f"  ⚠️  Failed to validate the CSV file: {e}")
                import traceback
                traceback.print_exc()
            
            metadata_path = None
            if args.metadata_path:
                metadata_path = Path(args.metadata_path).resolve()
                if not metadata_path.exists():
                    print(f"  ⚠️  Warning: metadata file does not exist, will fall back to unsupervised validation: {metadata_path}")
                    metadata_path = None

            # Step 6: Deterministic feature validation (optional; enabled by default, falls back to unsupervised when metadata does not exist)
            if args.enable_feature_analysis:
                print(f"\nStep 6: Deterministic feature validation (round {round_num})")
                if not features_csv_path.exists():
                    print(f"  ⚠️  Warning: feature file does not exist, skipping validation")
                    print(f"    Required: {features_csv_path}")
                else:
                    try:
                        prior_registry_for_validation = prior_registry or {}
                        if not prior_registry_for_validation:
                            prior_registry_for_validation = {
                                "all_raw_feature_names": existing_feature_names,
                                "all_historical_feature_names": existing_feature_names,
                                "live_feature_ids": [],
                                "entries": [],
                                "feature_id_to_column": {},
                            }

                        validation_result = validation_executor.validate_round(
                            raw_features_csv=features_csv_path,
                            feature_plan=feature_plan,
                            prior_registry=prior_registry_for_validation,
                            metadata_path=metadata_path,
                            round_dir=round_results_dir,
                            run_dir=results_dir,
                        )
                        round_validation_payload = {
                            "summary": validation_result.summary,
                            "decisions": [
                                {
                                    "feature_name": decision.feature_name,
                                    "status": decision.status,
                                    "reason_codes": decision.reason_codes,
                                    "compared_feature_ids": decision.compared_feature_ids,
                                    "validation_score": decision.validation_score,
                                }
                                for decision in validation_result.decisions
                            ],
                        }
                        previous_features_summary = {
                            "round": round_num,
                            "features_extracted": len(features),
                            "feature_names": [f.get('name', '') for f in features],
                            "validation_summary": validation_result.summary,
                            "planner_feedback": validation_result.planner_feedback,
                            "features_csv_path": str(features_csv_path),
                            "retained_features_csv_path": str(retained_features_csv_path) if retained_features_csv_path.exists() else None,
                            "total_features_so_far": _count_feature_columns(retained_features_csv_path),
                            "all_historical_feature_names": validation_result.planner_feedback.get("all_historical_feature_names", []),
                        }
                        cumulative_analysis_results.append({
                            "round": round_num,
                            "summary": validation_result.summary,
                            "planner_feedback": validation_result.planner_feedback,
                        })
                        print(f"  ✅ Deterministic validation complete")
                        print(f"     Retained {len(validation_result.retained_feature_names)} new features, dropped {len(validation_result.dropped_feature_names)} new features")
                    except Exception as e:
                        print(f"  ⚠️  Deterministic validation failed to execute: {e}")
                        import traceback
                        traceback.print_exc()
                        if features:
                            historical_names = prior_registry.get("all_historical_feature_names", existing_feature_names)
                            previous_features_summary = {
                                "round": round_num,
                                "features_extracted": len(features),
                                "feature_names": [f.get('name', '') for f in features],
                                "validation_summary": None,
                                "planner_feedback": _build_basic_planner_feedback(
                                    [f.get('name', '') for f in features],
                                    historical_names,
                                ),
                                "features_csv_path": str(features_csv_path),
                                "retained_features_csv_path": str(retained_features_csv_path) if retained_features_csv_path.exists() else None,
                                "total_features_so_far": _count_feature_columns(retained_features_csv_path),
                                "all_historical_feature_names": historical_names,
                            }
                            print(f"  ✅ Saved basic validation summary (for the next round of feature planning)")
            else:
                # No feature validation performed: only build a basic summary to help the next round avoid duplicates
                if features:
                    historical_names = prior_registry.get("all_historical_feature_names", existing_feature_names)
                    previous_features_summary = {
                        "round": round_num,
                        "features_extracted": len(features),
                        "feature_names": [f.get('name', '') for f in features],
                        "validation_summary": None,
                        "planner_feedback": _build_basic_planner_feedback(
                            [f.get('name', '') for f in features],
                            historical_names,
                        ),
                        "features_csv_path": str(features_csv_path),
                        "retained_features_csv_path": str(retained_features_csv_path) if retained_features_csv_path.exists() else None,
                        "total_features_so_far": _count_feature_columns(retained_features_csv_path if retained_features_csv_path.exists() else features_csv_path),
                        "all_historical_feature_names": historical_names,
                    }
                    print(f"  ⏭️  Skipped deterministic feature validation (--disable-feature-analysis); saved a basic summary for deduplication in the next round")
        
        # Save the results of this round
        round_results_path = round_results_dir / "round_results.json"
        round_output = {
            "round": round_num,
            "feature_plan": feature_plan,
            "sample_results": all_results,
            "validation": round_validation_payload,
        }
        with open(round_results_path, 'w', encoding='utf-8') as f:
            json.dump(round_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Round {round_num} complete!")
        print(f"   Results directory: {round_results_dir}")
        print(f"   Cumulative number of features: {len(pd.read_csv(features_csv_path).columns) - 1 if features_csv_path.exists() else 0}")
    
    # Summary after all rounds are complete
    print(f"\n{'='*80}")
    print(f"🎉 All {args.num_rounds} rounds of feature extraction complete!")
    print(f"{'='*80}")
    print(f"Final feature file: {features_csv_path}")
    if features_csv_path.exists():
        final_df = pd.read_csv(features_csv_path)
        print(f"Total number of features: {len(final_df.columns) - 1}")
        print(f"Total number of samples: {len(final_df)}")
    print(f"Results directory: {results_dir}")


if __name__ == "__main__":
    main()
