"""VLM Client.

Public build: the default and recommended path is the OpenAI-compatible
multimodal **API** client (``OnlineVLMClient``), which only needs the ``openai``
package. A local (self-hosted) Qwen3-VL client is still provided for advanced
users, but its heavy dependencies (``modelscope`` / ``transformers`` weights)
are imported lazily and are NOT installed by the default environment.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re
import time
from PIL import Image
import numpy as np

# ``torch`` is optional for the API-only path. It is available in the default
# environment (Cellpose-SAM depends on it), but we import it tolerantly so the
# agent still runs on machines without a local deep-learning stack.
try:
    import torch
except Exception:  # pragma: no cover - torch missing in a minimal API-only env
    torch = None

# Global VLM client instance (singleton pattern, used for a single GPU)
_global_vlm_client = None

# Client dict for multi-GPU mode (key is the GPU ID, value is a VLMClient instance)
_multi_gpu_clients: Dict[int, "VLMClient"] = {}

# Online API VLM client singleton (used when provider=online)
_global_online_vlm_client: Optional["OnlineVLMClient"] = None


def _repro_vlm_model_name() -> str:
    from config import settings
    if getattr(settings, "vlm_api_provider", "qwen") == "online":
        return settings.vlm_online_model or "online"
    return settings.vlm_model_path or "local-qwen"


def _repro_vlm_cache_lookup(
    mode: str,
    feature_names: List[str],
    image_paths: List[str],
    prompt: str,
    log_file: Optional[Path],
) -> Optional[Tuple[float, str, Optional[Dict[str, float]]]]:
    from config import settings
    from utils_modules.reproducibility import vlm_cache_get, vlm_cache_key
    if not settings.reproduce_mode:
        return None
    key = vlm_cache_key(
        mode=mode,
        feature_names=feature_names,
        image_paths=image_paths,
        prompt=prompt,
        model=_repro_vlm_model_name(),
        provider=settings.vlm_api_provider,
    )
    cached = vlm_cache_get(settings.reproduce_cache_dir, key)
    if cached and log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[VLM Repro] cache HIT ({mode}, key={key[:12]}…)\n")
            f.flush()
    return cached


def _repro_vlm_cache_store(
    mode: str,
    feature_names: List[str],
    image_paths: List[str],
    prompt: str,
    score: float,
    response: str,
    batch_scores: Optional[Dict[str, float]] = None,
) -> None:
    from config import settings
    from utils_modules.reproducibility import vlm_cache_key, vlm_cache_set
    if not settings.reproduce_mode:
        return
    key = vlm_cache_key(
        mode=mode,
        feature_names=feature_names,
        image_paths=image_paths,
        prompt=prompt,
        model=_repro_vlm_model_name(),
        provider=settings.vlm_api_provider,
    )
    vlm_cache_set(
        settings.reproduce_cache_dir,
        key,
        score=score,
        response=response,
        batch_scores=batch_scores,
    )


def get_vlm_client(model_path: Optional[str] = None, device: Optional[str] = None, gpu_id: Optional[int] = None) -> "VLMClient":
    """Get a VLM client instance
    
    Args:
        model_path: ModelScope model path (defaults to reading from config)
        device: device (cuda/cpu, defaults to reading from config)
        gpu_id: GPU ID (used for multi-GPU mode; if specified, returns the client for that GPU)
        
    Returns:
        VLMClient instance
    """
    from config import settings
    global _global_vlm_client, _multi_gpu_clients, _global_online_vlm_client
    
    # Online API mode (OpenAI-compatible multimodal): do not load any local model, ignore gpu_id, use a singleton
    if getattr(settings, "vlm_api_provider", "qwen") == "online":
        if _global_online_vlm_client is None:
            _global_online_vlm_client = OnlineVLMClient()
        return _global_online_vlm_client
    
    # Multi-GPU mode: create a separate client for each GPU
    if gpu_id is not None:
        if gpu_id not in _multi_gpu_clients:
            model_path = model_path or settings.vlm_model_path
            # Build the device string, e.g. "cuda:0", "cuda:1"
            if device is None:
                device = f"cuda:{gpu_id}"
            else:
                # If a device is provided, make sure the correct GPU ID is used
                if "cuda" in device.lower():
                    device = f"cuda:{gpu_id}"
            _multi_gpu_clients[gpu_id] = VLMClient(model_path=model_path, device=device, gpu_id=gpu_id)
            # Load the model immediately
            _multi_gpu_clients[gpu_id]._load_model()
        return _multi_gpu_clients[gpu_id]
    
    # Single-GPU mode: use the global singleton
    if _global_vlm_client is None:
        model_path = model_path or settings.vlm_model_path
        device = device or settings.vlm_device
        _global_vlm_client = VLMClient(model_path=model_path, device=device)
        # Load the model immediately
        _global_vlm_client._load_model()
    return _global_vlm_client


class VLMClient:
    """VLM client - based on Qwen3-VL-8B-Instruct"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, gpu_id: Optional[int] = None):
        """Initialize the VLM client
        
        Args:
            model_path: ModelScope model path (defaults to reading from config)
            device: device (cuda/cpu, defaults to reading from config)
            gpu_id: GPU ID (used for multi-GPU mode, specifies which GPU card to use)
        """
        from config import settings
        self.model_path = model_path or settings.vlm_model_path
        self.device = device or settings.vlm_device
        self.gpu_id = gpu_id
        self._model = None
        self._processor = None
        self._temp_dirs = []  # Track temporary directories, used for cleanup
    
    def _load_model(self):
        """Lazily load the local model (to avoid loading it at import time).

        The public build does not install local VLM dependencies by default; this code
        is only reached when the user explicitly chooses the local Qwen3-VL
        (--vlm-api-provider qwen), in which case ``modelscope`` and a GPU build of
        ``torch`` must be installed manually.
        """
        if self._model is None:
            try:
                from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
            except ImportError as exc:
                raise ImportError(
                    "The local VLM (Qwen3-VL) requires `modelscope`, but it is not installed in the current environment.\n"
                    "The public build uses the online API VLM by default: please switch to `--vlm-api-provider online`,\n"
                    "or manually install the local VLM dependencies: `pip install modelscope transformers accelerate` "
                    "and prepare a GPU build of PyTorch."
                ) from exc
            if torch is None:
                raise ImportError(
                    "The local VLM (Qwen3-VL) requires a GPU build of PyTorch, but `torch` is not installed in the current environment.\n"
                    "Please switch to the online API VLM: `--vlm-api-provider online`."
                )
            gpu_info = f" on GPU {self.gpu_id}" if self.gpu_id is not None else ""
            print(f"[VLM] Loading model: {self.model_path}{gpu_info}")
            
            # If a GPU ID is specified, use device_map to set the device
            if self.gpu_id is not None:
                device_map = f"cuda:{self.gpu_id}"
            else:
                device_map = "auto"
            
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            ).eval()
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print(f"[VLM] Model loading completed{gpu_info}")
    
    def score_feature(
        self,
        feature_def: Dict[str, Any],
        image_paths: List[str],
        full_prompt: str,
        max_images: Optional[int] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[float, str]:
        """Score a feature
        
        Args:
            feature_def: feature definition dict
            image_paths: list of image paths (local PNG file paths)
            full_prompt: the complete prompt (filled from a template, containing dataset information, knowledge sources, etc.)
            max_images: maximum number of images (to avoid exceeding the model's limit)
            
        Returns:
            (score, full_response) tuple; score is a float from 0-100, full_response is the complete response text
        """
        self._load_model()
        
        # Limit the number of images
        if len(image_paths) > max_images:
            msg = f"[VLM] Warning: the number of images ({len(image_paths)}) exceeds the limit ({max_images}); only using the first {max_images}"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            else:
                print(msg)
            image_paths = image_paths[:max_images]
        
        # Preprocess images: resize so the longer side does not exceed 512
        processed_image_paths = self._preprocess_images(image_paths, log_file)

        feat_name = feature_def.get("name", "")
        cached = _repro_vlm_cache_lookup(
            "single", [feat_name], image_paths, full_prompt, log_file
        )
        if cached is not None:
            score, full_response, _ = cached
            if hasattr(self, "_temp_dirs") and self._temp_dirs:
                self.cleanup_temp_files()
            return score, full_response
        
        # Record image information to the log
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM] Image preprocessing completed, {len(processed_image_paths)} images total\n")
                for i, img_path in enumerate(processed_image_paths[:5]):  # Only record the first 5
                    try:
                        img = Image.open(img_path)
                        f.write(f"[VLM] Image {i+1}: {Path(img_path).name}, mode: {img.mode}, size: {img.size}\n")
                        img.close()
                    except Exception as e:
                        f.write(f"[VLM] Image {i+1}: {Path(img_path).name}, failed to read: {e}\n")
                if len(processed_image_paths) > 5:
                    f.write(f"[VLM] ... and {len(processed_image_paths) - 5} more images\n")
                f.flush()
        
        # Ensure the image paths are absolute
        image_paths_abs = [str(Path(p).resolve()) for p in processed_image_paths]
        
        # Build the message (Qwen3-VL format)
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths_abs],
                {"type": "text", "text": full_prompt},
            ],
        }]
        
        # Prepare the input
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM] Preparing input, number of image paths: {len(image_paths_abs)}\n")
                f.write(f"[VLM] First 3 image paths: {[Path(p).name for p in image_paths_abs[:3]]}\n")
                f.flush()
        
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Record shape information of the input tensors
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM] Input preparation completed\n")
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            f.write(f"[VLM] Input key '{key}': shape={value.shape}, dtype={value.dtype}\n")
                    f.flush()
        except Exception as e:
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM] Error: failed to prepare input: {e}\n")
                    import traceback
                    f.write(f"[VLM] Traceback:\n{traceback.format_exc()}\n")
                    f.flush()
            raise
        
        # Move to the correct device
        device = next(self._model.parameters()).device
        device_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        # Clean up the original inputs (may contain tensors on the CPU)
        del inputs
        
        # Generate the response (do not print to the console, to avoid interrupting the progress bar)
        msg = f"[VLM] Generating score (using {len(image_paths_abs)} images)..."
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
                f.flush()  # Ensure it is written immediately
        # Do not print to the console, to avoid interrupting the progress bar
        
        from config import settings
        import signal
        
        # Record the start time, used to detect whether it is stuck
        start_time = time.time()
        timeout_seconds = 60  # 60-second timeout
        
        # Define the timeout handler function
        def timeout_handler(signum, frame):
            raise TimeoutError(f"VLM generation timeout after {timeout_seconds} seconds")
        
        try:
            # Set the timeout (only supported on Unix systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **device_inputs, 
                    max_new_tokens=settings.vlm_max_tokens,  # Read the token limit from config
                    do_sample=False,
                    temperature=__import__("config").get_vlm_temperature()
                )
            
            # Cancel the timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            elapsed = time.time() - start_time
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM] Generation completed, elapsed: {elapsed:.2f}s\n")
                    f.flush()
        except TimeoutError as e:
            elapsed = time.time() - start_time
            error_msg = f"[VLM] Generation timeout after {elapsed:.2f}s (limit: {timeout_seconds}s)"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.flush()
            # Clean up GPU memory
            del device_inputs
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise TimeoutError(error_msg)
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"[VLM] Generation failed (elapsed {elapsed:.2f}s): {str(e)}"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    import traceback
                    f.write(traceback.format_exc() + "\n")
                    f.flush()
            # Clean up GPU memory
            if 'device_inputs' in locals():
                del device_inputs
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise
        finally:
            # Ensure the timeout is cancelled
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        # Extract the generated text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(device_inputs["input_ids"], generated_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        full_response = output_text[0] if output_text else ""
        score = self._parse_score(full_response, log_file)
        
        # Clean up GPU memory: delete all intermediate tensors and variables
        del generated_ids, generated_ids_trimmed, device_inputs, output_text, messages, image_paths_abs, processed_image_paths
        # Force garbage collection and clear the GPU cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        
        # Clean up temporary files (if any)
        if hasattr(self, '_temp_dirs') and self._temp_dirs:
            self.cleanup_temp_files()
        
        # Only print a brief result; detailed information is written to the log
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM] Scoring completed: {score:.2f}\n")
                f.write(f"[VLM] Full response:\n{full_response}\n\n")
        # Do not print to the console, to avoid interrupting the progress bar (detailed information is already written to the log file)
        
        _repro_vlm_cache_store(
            "single", [feat_name], image_paths, full_prompt, score, full_response
        )
        return score, full_response
    
    def score_features_batch(
        self,
        features: List[Dict[str, Any]],
        image_paths: List[str],
        full_prompt: str,
        max_images: Optional[int] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[Dict[str, float], str]:
        """Score multiple features in a batch (process all features in one call)
        
        Args:
            features: list of feature definitions
            image_paths: list of image paths (local PNG file paths)
            full_prompt: the complete prompt (filled from a template, containing dataset information, knowledge sources, etc.)
            max_images: maximum number of images (to avoid exceeding the model's limit)
            log_file: path to the log file (optional)
            
        Returns:
            (scores_dict, full_response) tuple; scores_dict maps feature names to scores, full_response is the complete response text
        """
        self._load_model()
        
        # Limit the number of images
        if len(image_paths) > max_images:
            msg = f"[VLM Batch] Warning: the number of images ({len(image_paths)}) exceeds the limit ({max_images}); only using the first {max_images}"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            else:
                print(msg)
            image_paths = image_paths[:max_images]
        
        # Preprocess images: resize so the longer side does not exceed 512
        processed_image_paths = self._preprocess_images(image_paths, log_file)

        feat_names = [f.get("name", "") for f in features]
        cached = _repro_vlm_cache_lookup(
            "batch", feat_names, image_paths, full_prompt, log_file
        )
        if cached is not None:
            _, full_response, batch_scores = cached
            if batch_scores is not None:
                if hasattr(self, "_temp_dirs") and self._temp_dirs:
                    self.cleanup_temp_files()
                return batch_scores, full_response
        
        # Record image information to the log
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM Batch] Image preprocessing completed, {len(processed_image_paths)} images total\n")
                f.flush()
        
        # Ensure the image paths are absolute
        image_paths_abs = [str(Path(p).resolve()) for p in processed_image_paths]
        
        # Build the message (Qwen3-VL format)
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": path} for path in image_paths_abs],
                {"type": "text", "text": full_prompt},
            ],
        }]
        
        # Prepare the input
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM Batch] Preparing input, number of image paths: {len(image_paths_abs)}\n")
                f.write(f"[VLM Batch] Number of features to process: {len(features)}\n")
                f.flush()
        
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Record shape information of the input tensors
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM Batch] Input preparation completed\n")
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            f.write(f"[VLM Batch] Input key '{key}': shape={value.shape}, dtype={value.dtype}\n")
                    f.flush()
        except Exception as e:
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM Batch] Error: failed to prepare input: {e}\n")
                    import traceback
                    f.write(f"[VLM Batch] Traceback:\n{traceback.format_exc()}\n")
                    f.flush()
            raise
        
        # Move to the correct device
        device = next(self._model.parameters()).device
        device_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in inputs.items()
        }
        
        # Clean up the original inputs
        del inputs
        
        # Generate the response
        msg = f"[VLM Batch] Generating batch scores (using {len(image_paths_abs)} images, {len(features)} features)..."
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
                f.flush()
        
        from config import settings
        import signal
        
        # Record the start time
        start_time = time.time()
        timeout_seconds = 120  # Batch processing may take longer, set to 120 seconds
        
        # Define the timeout handler function
        def timeout_handler(signum, frame):
            raise TimeoutError(f"VLM batch generation timeout after {timeout_seconds} seconds")
        
        try:
            # Set the timeout (only supported on Unix systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **device_inputs, 
                    max_new_tokens=settings.vlm_max_tokens,
                    do_sample=False,
                    temperature=__import__("config").get_vlm_temperature()
                )
            
            # Cancel the timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            elapsed = time.time() - start_time
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[VLM Batch] Generation completed, elapsed: {elapsed:.2f}s\n")
                    f.flush()
        except TimeoutError as e:
            elapsed = time.time() - start_time
            error_msg = f"[VLM Batch] Generation timeout after {elapsed:.2f}s (limit: {timeout_seconds}s)"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.flush()
            # Clean up GPU memory
            del device_inputs
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise TimeoutError(error_msg)
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"[VLM Batch] Generation failed (elapsed {elapsed:.2f}s): {str(e)}"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    import traceback
                    f.write(traceback.format_exc() + "\n")
                    f.flush()
            # Clean up GPU memory
            if 'device_inputs' in locals():
                del device_inputs
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            raise
        finally:
            # Ensure the timeout is cancelled
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        
        # Extract the generated text
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(device_inputs["input_ids"], generated_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        full_response = output_text[0] if output_text else ""
        scores_dict = self._parse_scores_batch(full_response, features, log_file)
        
        # Clean up GPU memory
        del generated_ids, generated_ids_trimmed, device_inputs, output_text, messages, image_paths_abs, processed_image_paths
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean up temporary files
        if hasattr(self, '_temp_dirs') and self._temp_dirs:
            self.cleanup_temp_files()
        
        # Record the results to the log
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM Batch] Batch scoring completed\n")
                for feat_name, score in scores_dict.items():
                    f.write(f"[VLM Batch] {feat_name}: {score:.2f}\n")
                f.write(f"[VLM Batch] Full response:\n{full_response}\n\n")
        
        mean_score = float(sum(scores_dict.values()) / len(scores_dict)) if scores_dict else 0.0
        _repro_vlm_cache_store(
            "batch", feat_names, image_paths, full_prompt,
            mean_score, full_response, batch_scores=scores_dict,
        )
        return scores_dict, full_response
    
    def _parse_scores_batch(self, text: str, features: List[Dict[str, Any]], log_file: Optional[Path] = None) -> Dict[str, float]:
        """Parse the scores of multiple features from text
        
        Args:
            text: text generated by the VLM
            features: list of feature definitions
            log_file: path to the log file (optional)
            
        Returns:
            dict mapping feature names to scores
        """
        scores_dict = {}
        feature_names = [feat.get("name", "unknown") for feat in features]
        
        # Method 1: look for JSON format {"feature_name": score, ...}
        try:
            # Try to parse the whole text as JSON
            data = json.loads(text.strip())
            if isinstance(data, dict):
                for feat_name in feature_names:
                    if feat_name in data:
                        score = float(data[feat_name])
                        if 0 <= score <= 100:
                            scores_dict[feat_name] = score
                if len(scores_dict) == len(features):
                    return scores_dict
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        
        # Method 2: look for multi-line JSON format, one feature per line
        lines = text.strip().split('\n')
        for line in lines:
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict):
                    for feat_name in feature_names:
                        if feat_name in data:
                            score = float(data[feat_name])
                            if 0 <= score <= 100:
                                scores_dict[feat_name] = score
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        # Method 3: look for the score of each feature name (using regular expressions)
        for feat_name in feature_names:
            if feat_name in scores_dict:
                continue  # Already found
            
            # Look for {"feature_name": score} or "feature_name": score
            patterns = [
                rf'"{re.escape(feat_name)}"\s*:\s*([0-9]+\.?[0-9]*)',
                rf"'{re.escape(feat_name)}'\s*:\s*([0-9]+\.?[0-9]*)",
                rf'{re.escape(feat_name)}\s*:\s*([0-9]+\.?[0-9]*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 0 <= score <= 100:
                            scores_dict[feat_name] = score
                            break
                    except (ValueError, IndexError):
                        continue
        
        # Method 4: if not all scores have been found, try to find numbers in the last few lines
        # Assume the VLM outputs the scores in feature order
        if len(scores_dict) < len(features):
            numbers = re.findall(r'([0-9]+\.?[0-9]*)', text)
            # Only take numbers between 0 and 100
            valid_scores = []
            for num_str in numbers:
                try:
                    score = float(num_str)
                    if 0 <= score <= 100:
                        valid_scores.append(score)
                except ValueError:
                    continue
            
            # If the number of found numbers equals the number of features, assign them in order
            if len(valid_scores) == len(features):
                for i, feat_name in enumerate(feature_names):
                    if feat_name not in scores_dict:
                        scores_dict[feat_name] = valid_scores[i]
        
        # For features whose score was not found, return 0.0 and log a warning
        for feat_name in feature_names:
            if feat_name not in scores_dict:
                msg = f"[VLM Batch] Warning: unable to parse the score for feature '{feat_name}' from the response; returning the default value 0.0"
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(msg + "\n")
                else:
                    print(msg)
                scores_dict[feat_name] = 0.0
        
        return scores_dict
    
    def _parse_score(self, text: str, log_file: Optional[Path] = None) -> float:
        """Parse the score from text
        
        Prefer looking for a JSON-format score, then look for a number
        
        Args:
            text: text generated by the VLM
            
        Returns:
            a float score between 0 and 100
        """
        # Method 1: look for JSON format {"score": <float>}
        json_patterns = [
            r'\{[^}]*"score"\s*:\s*([0-9]+\.?[0-9]*)[^}]*\}',  # Standard JSON
            r'\{[^}]*"score"\s*:\s*([0-9]+\.?[0-9]*)\s*\}',  # Strict format
            r'"score"\s*:\s*([0-9]+\.?[0-9]*)',  # Simplified format
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure the score is within the valid range
                    if 0 <= score <= 100:
                        return score
                except (ValueError, IndexError):
                    continue
        
        # Method 2: look for a number in the last line (usually the line containing the JSON)
        lines = text.strip().split('\n')
        for line in reversed(lines):
            # Try to parse the whole line as JSON
            try:
                data = json.loads(line.strip())
                if isinstance(data, dict) and "score" in data:
                    score = float(data["score"])
                    if 0 <= score <= 100:
                        return score
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
            
            # Look for a number
            number_match = re.search(r'([0-9]+\.?[0-9]*)', line)
            if number_match:
                try:
                    score = float(number_match.group(1))
                    if 0 <= score <= 100:
                        return score
                except ValueError:
                    continue
        
        # Method 3: look for a number between 0 and 100 in the entire text
        all_numbers = re.findall(r'([0-9]+\.?[0-9]*)', text)
        for num_str in reversed(all_numbers):  # From back to front, preferring the last number
            try:
                score = float(num_str)
                if 0 <= score <= 100:
                    return score
            except ValueError:
                continue
        
        # If nothing is found, return 0.0 and log a warning
        msg = f"[VLM] Warning: unable to parse a score from the response; returning the default value 0.0"
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
                print_len = min(500, len(text))
                f.write(f"[VLM] Response text (last {print_len} characters): {text[-print_len:]}\n")
        else:
            print(msg)
            print_len = min(500, len(text))
            print(f"[VLM] Response text (last {print_len} characters): {text[-print_len:]}")
        return 0.0
    
    def _preprocess_images(self, image_paths: List[str], log_file: Optional[Path] = None) -> List[str]:
        """Preprocess images: resize so the longer side does not exceed the configured maximum size
        
        Args:
            image_paths: list of original image paths
            
        Returns:
            list of processed image paths (if an image needs resizing, it is saved to a temporary file)
        """
        from config import settings
        import tempfile
        import atexit
        import shutil
        
        max_size = settings.vlm_image_resize_max
        processed_paths = []
        temp_dir = None
        temp_dirs_to_cleanup = []  # Track temporary directories that need cleanup
        
        for img_path in image_paths:
            try:
                # Open the image
                img = Image.open(img_path)
                width, height = img.size
                img_mode = img.mode
                
                # Record image information to the log
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[VLM] Preprocessing image: {Path(img_path).name}, mode: {img_mode}, size: {width}x{height}\n")
                        f.flush()
                
                # Check whether the image size is abnormal (it may be an incorrect slice file)
                if width < 10 or height < 10 or (width == 3 and height == 512) or (width == 512 and height == 3):
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[VLM] Error: image {Path(img_path).name} has an abnormal size: {width}x{height}, mode: {img_mode}\n")
                            f.write(f"[VLM] This may be an old, incorrectly formatted slice file; consider deleting the slices directory and regenerating\n")
                            f.flush()
                    img.close()
                    # Skip this abnormal image and use the next one
                    continue
                
                # Ensure the image is in RGB mode (if not, convert to RGB)
                if img_mode not in ['RGB', 'RGBA']:
                    if log_file:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[VLM] Warning: image {Path(img_path).name} has mode {img_mode}; converting to RGB\n")
                            f.flush()
                    # Convert to RGB
                    if img_mode == 'L':
                        # Grayscale image: copy to the three RGB channels
                        rgb_img = Image.new('RGB', img.size)
                        rgb_img.paste(img, (0, 0))
                        img.close()
                        img = rgb_img
                        img_mode = 'RGB'
                    elif img_mode == 'P':
                        # Palette mode: convert to RGB
                        img = img.convert('RGB')
                        img_mode = 'RGB'
                    else:
                        # Other modes: try to convert
                        img = img.convert('RGB')
                        img_mode = 'RGB'
                
                # Check whether resizing is needed (the longer side exceeds the configured maximum size)
                max_dim = max(width, height)
                if max_dim <= max_size:
                    # No resize needed, but if the mode changed, it needs to be saved
                    # Reopen the file to get the original mode (since img may already have been converted)
                    try:
                        original_img = Image.open(img_path)
                        original_mode = original_img.mode
                        original_img.close()
                    except:
                        original_mode = img_mode
                    
                    if img_mode != original_mode:
                        # The converted image needs to be saved
                        if temp_dir is None:
                            temp_dir = Path(tempfile.mkdtemp(prefix="vlm_images_"))
                            temp_dirs_to_cleanup.append(temp_dir)
                            def cleanup_temp_dir():
                                if temp_dir.exists():
                                    try:
                                        shutil.rmtree(temp_dir)
                                    except:
                                        pass
                            atexit.register(cleanup_temp_dir)
                        
                        img_name = Path(img_path).name
                        temp_path = temp_dir / f"converted_{img_name}"
                        img.save(temp_path, format="PNG")
                        if log_file:
                            with open(log_file, 'a', encoding='utf-8') as f:
                                f.write(f"[VLM] Saved converted image: {temp_path.name} ({original_mode} -> {img_mode})\n")
                                f.flush()
                        processed_paths.append(str(temp_path))
                    else:
                        processed_paths.append(img_path)
                    img.close()  # Explicitly close the image
                    continue
                
                # Resize needed: scale proportionally so the longer side is the configured maximum size
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                # Resize the image
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img.close()  # Close the original image
                
                # Record the resize information
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[VLM] Resized image: {Path(img_path).name} {width}x{height} -> {new_width}x{new_height}\n")
                        f.flush()
                
                # Save to a temporary file
                if temp_dir is None:
                    temp_dir = Path(tempfile.mkdtemp(prefix="vlm_images_"))
                    temp_dirs_to_cleanup.append(temp_dir)
                    # Register the cleanup function (cleaned up when the program exits)
                    def cleanup_temp_dir():
                        if temp_dir.exists():
                            try:
                                shutil.rmtree(temp_dir)
                            except:
                                pass
                    atexit.register(cleanup_temp_dir)
                
                # Generate the temporary file path
                img_name = Path(img_path).name
                temp_path = temp_dir / f"resized_{img_name}"
                img_resized.save(temp_path, format="PNG")
                
                # Verify the saved image
                if log_file:
                    try:
                        verify_img = Image.open(temp_path)
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[VLM] Verifying saved image: {temp_path.name}, mode: {verify_img.mode}, size: {verify_img.size}\n")
                            f.flush()
                        verify_img.close()
                    except Exception as e:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[VLM] Warning: failed to verify the saved image: {e}\n")
                            f.flush()
                
                img_resized.close()  # Close the resized image
                processed_paths.append(str(temp_path))
                
            except Exception as e:
                # If processing fails, log the detailed error and skip
                msg = f"[VLM] Error: failed to process image {img_path}: {e}"
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(msg + "\n")
                        import traceback
                        f.write(f"[VLM] Traceback: {traceback.format_exc()}\n")
                        f.flush()
                else:
                    print(msg)
                # Do not add the failed image; skip it
                if 'img' in locals():
                    try:
                        img.close()
                    except:
                        pass
                continue
        
        # Save the list of temporary directories to an instance variable for later cleanup
        if temp_dirs_to_cleanup:
            if not hasattr(self, '_temp_dirs'):
                self._temp_dirs = []
            self._temp_dirs.extend(temp_dirs_to_cleanup)
        
        # Record the final processing result
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM] Image preprocessing completed: {len(image_paths)} input, {len(processed_paths)} successfully processed\n")
                if len(processed_paths) == 0:
                    f.write(f"[VLM] Warning: no images were successfully processed!\n")
                f.flush()
        
        if len(processed_paths) == 0:
            raise ValueError(f"No images were successfully processed; please check the image file format. Original paths: {image_paths[:3]}")
        
        return processed_paths
    
    def cleanup_temp_files(self):
        """Clean up all temporary file directories"""
        import shutil
        for temp_dir in self._temp_dirs:
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    pass  # Ignore cleanup errors
        self._temp_dirs.clear()


class OnlineVLMClient(VLMClient):
    """Online API VLM client - called through an OpenAI-compatible multimodal interface (e.g. GPT-4o and other multimodal models).

    Inherits from VLMClient to reuse the already-validated image preprocessing (_preprocess_images) and score parsing
    (_parse_score / _parse_scores_batch), but does not load any local model; instead it sends images to the online model
    as base64 data-URLs. The public interface (score_feature / score_features_batch)
    is exactly the same as the local VLMClient, so the upstream _execute_vlm_features_batch does not need to change.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        from config import settings
        self.base_url = base_url or settings.vlm_online_base_url
        self.api_key = api_key or settings.vlm_online_api_key
        self.model = model or settings.vlm_online_model
        self.default_headers = default_headers if default_headers is not None else settings.vlm_online_default_headers
        self.gpu_id = None
        self._temp_dirs = []
        self._client = None
        self._v1_fallback_tried = False

    def _load_model(self):
        """Online mode does not need to load a local model; lazily create the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            kwargs = {"base_url": self.base_url, "api_key": self.api_key}
            if self.default_headers:
                kwargs["default_headers"] = self.default_headers
            self._client = OpenAI(**kwargs)
            print(f"[VLM-Online] Initialized the online client: base_url={self.base_url}, model={self.model}")

    def _maybe_switch_to_v1_base_url(self, exc: Exception, log_fn) -> bool:
        from utils_modules.openai_base_url import is_http_404_error, with_v1_suffix

        if self._v1_fallback_tried or not is_http_404_error(exc):
            return False
        candidate = with_v1_suffix(str(self.base_url or ""))
        if not candidate:
            return False
        self._v1_fallback_tried = True
        self.base_url = candidate
        self._client = None
        log_fn(f"[VLM-Online] API returned 404; retrying with base_url={candidate}")
        try:
            from config import settings as _settings
            _settings.vlm_online_base_url = candidate
        except Exception:
            pass
        self._load_model()
        return True

    def _images_to_content(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Convert local images to valid PNG data URLs for the online API.

        Do not infer the MIME type from the filename.  Scientific images are
        commonly TIFF files, and ``_preprocess_images`` may return the original
        file when it does not need resizing.  Labelling those TIFF bytes as
        ``image/png`` makes OpenAI-compatible endpoints reject the request with
        ``invalid_image_format``.
        """
        import base64
        import io

        content = []
        errors = []
        for p in image_paths:
            try:
                with Image.open(p) as img:
                    # PNG supports RGB/RGBA/L, which covers the modes normally
                    # produced by preprocessing. Convert uncommon scientific
                    # modes (I;16, F, CMYK, palette, etc.) deterministically.
                    if img.mode not in ("RGB", "RGBA", "L"):
                        img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")

                image_bytes = buffer.getvalue()
                if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("PNG encoding did not produce a valid signature")

                b64 = base64.b64encode(image_bytes).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except Exception as exc:
                errors.append(f"{p}: {exc}")

        if not content:
            details = "; ".join(errors) or "no image paths were provided"
            raise ValueError(f"Unable to prepare any valid image for the VLM: {details}")
        return content

    def _chat_with_retry(self, content: List[Dict[str, Any]], log_file: Optional[Path] = None) -> str:
        """Call the online multimodal interface and return the text response, with retries + a hard wall-clock timeout.

        On top of httpx's own read timeout, add a thread-level hard timeout (wall-clock):
        even if the server's "trickling" slow response causes the read timeout to never fire, once the time is up it will forcibly abandon this attempt and retry,
        preventing a single sample from hanging the whole round of VLM scoring indefinitely. The thread-level timeout does not depend on main-thread signals and can be used from any thread.
        """
        import concurrent.futures

        from config import settings
        self._load_model()
        messages = [{"role": "user", "content": content}]
        last_exc = None
        max_attempts = max(1, getattr(settings, "vlm_online_max_attempts", 3))
        request_timeout = getattr(settings, "vlm_online_request_timeout", 150)
        # Hard timeout (wall-clock): take a bit of buffer above request_timeout, as a fallback for when the read-timeout fails
        hard_timeout = max(
            request_timeout + 30,
            getattr(settings, "vlm_online_hard_timeout", 180),
        )

        def _do_request() -> str:
            from config import get_vlm_temperature, settings as _settings
            req_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": settings.vlm_online_max_tokens,
                "temperature": get_vlm_temperature(),
                "timeout": request_timeout,
            }
            if _settings.reproduce_mode:
                req_kwargs["seed"] = _settings.reproduce_seed
            resp = self._client.chat.completions.create(**req_kwargs)
            return resp.choices[0].message.content or ""

        def _log(msg: str) -> None:
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
                    f.flush()
            else:
                print(msg)

        for attempt in range(1, max_attempts + 1):
            # Each attempt runs in a one-off single-thread executor, making it easy to abandon a hung call when the time is up
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_do_request)
            try:
                result = future.result(timeout=hard_timeout)
                executor.shutdown(wait=False)
                return result
            except concurrent.futures.TimeoutError:
                last_exc = TimeoutError(
                    f"VLM-Online hard timeout: a single request took more than {hard_timeout}s without returning (likely a slow/hung server response)"
                )
                _log(f"[VLM-Online] Hard timeout interruption (attempt {attempt}/{max_attempts}): no return after >{hard_timeout}s, abandoning this attempt and retrying")
                # Do not wait for the hung thread (the connection will close on its own with the httpx timeout), to avoid blocking
                executor.shutdown(wait=False)
            except Exception as exc:
                last_exc = exc
                _log(f"[VLM-Online] Request failed (attempt {attempt}/{max_attempts}): {type(exc).__name__}: {exc}")
                executor.shutdown(wait=False)
                if self._maybe_switch_to_v1_base_url(exc, _log):
                    # Immediate retry with corrected base_url (does not consume an extra backoff cycle).
                    try:
                        return _do_request()
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        _log(
                            f"[VLM-Online] Retry after /v1 base_url still failed: "
                            f"{type(retry_exc).__name__}: {retry_exc}"
                        )
            if attempt < max_attempts:
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
        raise last_exc if last_exc else RuntimeError("[VLM-Online] Unknown error")

    def score_feature(
        self,
        feature_def: Dict[str, Any],
        image_paths: List[str],
        full_prompt: str,
        max_images: Optional[int] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[float, str]:
        """Score a single feature (online API version; the interface matches the local one)."""
        if max_images and len(image_paths) > max_images:
            image_paths = image_paths[:max_images]
        processed = self._preprocess_images(image_paths, log_file)
        feat_name = feature_def.get("name", "")
        cached = _repro_vlm_cache_lookup(
            "single", [feat_name], image_paths, full_prompt, log_file
        )
        if cached is not None:
            score, full_response, _ = cached
            if self._temp_dirs:
                self.cleanup_temp_files()
            return score, full_response
        content = self._images_to_content([str(Path(p).resolve()) for p in processed])
        content.append({"type": "text", "text": full_prompt})

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM-Online] Single-feature scoring, number of images: {len(processed)}, model: {self.model}\n")
                f.flush()

        full_response = self._chat_with_retry(content, log_file)
        score = self._parse_score(full_response, log_file)

        if self._temp_dirs:
            self.cleanup_temp_files()

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM-Online] Scoring completed: {score:.2f}\n")
                f.write(f"[VLM-Online] Full response:\n{full_response}\n\n")
        _repro_vlm_cache_store(
            "single", [feat_name], image_paths, full_prompt, score, full_response
        )
        return score, full_response

    def score_features_batch(
        self,
        features: List[Dict[str, Any]],
        image_paths: List[str],
        full_prompt: str,
        max_images: Optional[int] = None,
        log_file: Optional[Path] = None
    ) -> Tuple[Dict[str, float], str]:
        """Score multiple features in a batch (online API version; the interface matches the local one)."""
        if max_images and len(image_paths) > max_images:
            msg = f"[VLM-Online Batch] The number of images ({len(image_paths)}) exceeds the limit ({max_images}); only using the first {max_images}"
            if log_file:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            else:
                print(msg)
            image_paths = image_paths[:max_images]

        processed = self._preprocess_images(image_paths, log_file)
        feat_names = [f.get("name", "") for f in features]
        cached = _repro_vlm_cache_lookup(
            "batch", feat_names, image_paths, full_prompt, log_file
        )
        if cached is not None:
            _, full_response, batch_scores = cached
            if batch_scores is not None:
                if self._temp_dirs:
                    self.cleanup_temp_files()
                return batch_scores, full_response
        content = self._images_to_content([str(Path(p).resolve()) for p in processed])
        content.append({"type": "text", "text": full_prompt})

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM-Online Batch] Number of images: {len(processed)}, number of features: {len(features)}, model: {self.model}\n")
                f.flush()

        start_time = time.time()
        full_response = self._chat_with_retry(content, log_file)
        elapsed = time.time() - start_time
        scores_dict = self._parse_scores_batch(full_response, features, log_file)

        if self._temp_dirs:
            self.cleanup_temp_files()

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[VLM-Online Batch] Batch scoring completed (elapsed {elapsed:.2f}s)\n")
                for feat_name, score in scores_dict.items():
                    f.write(f"[VLM-Online Batch] {feat_name}: {score:.2f}\n")
                f.write(f"[VLM-Online Batch] Full response:\n{full_response}\n\n")
        mean_score = float(sum(scores_dict.values()) / len(scores_dict)) if scores_dict else 0.0
        _repro_vlm_cache_store(
            "batch", feat_names, image_paths, full_prompt,
            mean_score, full_response, batch_scores=scores_dict,
        )
        return scores_dict, full_response
