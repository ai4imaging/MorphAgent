"""Code generation module - uses the LLM to generate feature extraction code"""
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings, make_chat_llm, get_code_temperature
from nodes.prompt_gen import load_template, fill_template
from state import AgentState


@dataclass
class CodeResult:
    """Code generation result"""
    code: Optional[str]
    prompt: str
    response: str
    planning_response: Optional[str] = None  # CoT planning response


class CodeGenerator:
    """Code generator"""
    
    def __init__(self):
        kwargs: Dict[str, Any] = {"temperature": get_code_temperature()}
        if settings.reproduce_mode:
            kwargs["seed"] = settings.reproduce_seed
        self.llm = make_chat_llm(**kwargs)
    
    def generate_planning(
        self,
        feature: Dict[str, Any],
        state: AgentState,
        data_statistics: Dict[str, Any]
    ) -> str:
        """Generate the code implementation plan (CoT step)
        
        Args:
            feature: feature definition dict
            state: agent state
            data_statistics: data statistics (statistics for the image and segmentation)
            
        Returns:
            The implementation plan text
        """
        # Load the code planning template
        template_dict = load_template("code_planning")
        template_str = template_dict.get("template", "")
        
        # Prepare the fill data
        fill_data = {
            "feature_name": feature.get("name", ""),
            "feature_description": feature.get("description", ""),
            "feature_category": feature.get("category", ""),
            "image_shape": data_statistics.get("image_shape", "Unknown"),
            "image_dtype": data_statistics.get("image_dtype", "Unknown"),
            "image_min": data_statistics.get("image_min", "Unknown"),
            "image_max": data_statistics.get("image_max", "Unknown"),
            "segmentation_files_info": data_statistics.get("segmentation_files_info", ""),
            "segmentation_statistics": data_statistics.get("segmentation_statistics", ""),
            "seg_keys_description": data_statistics.get("seg_keys_description", "No segmentation. Use def extract(img): only."),
            "channel_information": data_statistics.get("channel_information", state.get("channel_information", "")),
            "segmentation_mask_order": state.get("segmentation_mask_order", data_statistics.get("segmentation_mask_order", "")),
        }
        
        # Create a temporary state to fill the template
        temp_state = state.copy()
        temp_state.update(fill_data)
        
        # Fill the template (fill_template handles all placeholders)
        filled_prompt = fill_template(template_str, temp_state)
        
        # Call the LLM to generate the plan
        print(f"  [CoT Planning] Generating the implementation plan...")
        messages = [HumanMessage(content=filled_prompt)]
        try:
            response = self.llm.invoke(messages)
            planning_text = response.content if hasattr(response, 'content') else str(response)
            print(f"  [OK] Plan generation complete ({len(planning_text)} characters)")
            return planning_text
        except Exception as e:
            print(f"  [ERROR] Plan generation failed: {e}; skipping the CoT step")
            return ""
    
    def generate(
        self,
        feature: Dict[str, Any],
        state: AgentState,
        guidance_message: str = "",
        planning_text: Optional[str] = None
    ) -> CodeResult:
        """Generate feature extraction code
        
        Args:
            feature: feature definition dict
            state: agent state (includes the dataset description, knowledge sources, etc.)
            guidance_message: guidance message from code repair (optional)
            planning_text: the CoT-generated implementation plan (optional)
            
        Returns:
            A CodeResult object containing the generated code, prompt, and response
        """
        # Load the code generation template
        template_dict = load_template("code_generation")
        template_str = template_dict.get("template", "")
        
        # Get the data statistics from the state (if already collected)
        data_statistics = state.get("data_statistics", {})
        
        # Get the segmentation file information (scanned from the first sample or taken from the statistics)
        segmentation_files_info = data_statistics.get("segmentation_files_info", "")
        seg_keys_description_fallback = "No segmentation. Use def extract(img): only."
        if not segmentation_files_info:
            # If there are no statistics, try to scan
            sample_id = state.get("sample_id", "")
            if sample_id:
                from config import settings
                data_root = state.get("data_root")
                if not data_root:
                    image_paths = state.get("image_paths", [])
                    if image_paths:
                        first_image = Path(image_paths[0])
                        if "dataset" in first_image.parts:
                            dataset_idx = first_image.parts.index("dataset")
                            data_root = Path(*first_image.parts[:dataset_idx+1])
                        else:
                            data_root = first_image.parent.parent
                
                if data_root:
                    sample_dir = Path(data_root) / sample_id if isinstance(data_root, (str, Path)) else Path(str(data_root)) / sample_id
                    seg_dir = sample_dir / "segmentation"
                    if seg_dir.exists() and seg_dir.is_dir():
                        seg_files = []
                        for seg_file in sorted(seg_dir.glob("*.tif")):
                            seg_files.append({
                                "name": seg_file.name,
                                "stem": seg_file.stem,
                                "path": str(seg_file.relative_to(sample_dir))
                            })
                        
                        if seg_files:
                            segmentation_files_info = f"\n**Available Segmentation Files** (in `sample_dir/segmentation/`):\n"
                            for i, seg_file in enumerate(seg_files, 1):
                                segmentation_files_info += f"  {i}. `{seg_file['name']}` (stem: `{seg_file['stem']}`)\n"
                            segmentation_files_info += f"\nTotal: {len(seg_files)} segmentation file(s) available.\n"
                            keys_list = ", ".join(repr(s["stem"]) for s in seg_files)
                            seg_keys_description_fallback = (
                                f"**Segmentation keys for this dataset** (use these exact keys in `seg`): {keys_list}.\n"
                                "Access by key only, e.g. seg[\"mask_cell\"], seg.get(\"mask_nucleus\"). Do NOT guess by position or index."
                            )
        
        # Prepare the fill data (using the statistics)
        # Note: fill_template gets all required fields from the state; we only need to add the extra fields
        fill_data = {
            "feature_name": feature.get("name", ""),
            "feature_description": feature.get("description", ""),
            "feature_category": feature.get("category", ""),
            "feature_guidance_message": guidance_message if guidance_message else "",
            "segmentation_files_info": data_statistics.get("segmentation_files_info", segmentation_files_info),
            "seg_keys_description": data_statistics.get("seg_keys_description", seg_keys_description_fallback),
            "image_shape": data_statistics.get("image_shape", "Unknown"),
            "image_dtype": data_statistics.get("image_dtype", "Unknown"),
            "image_min": data_statistics.get("image_min", "Unknown"),
            "image_max": data_statistics.get("image_max", "Unknown"),
            "segmentation_statistics": data_statistics.get("segmentation_statistics", ""),
            "channel_information": data_statistics.get("channel_information", state.get("channel_information", "")),
            "segmentation_mask_order": state.get("segmentation_mask_order", data_statistics.get("segmentation_mask_order", "")),
        }
        
        # Create a temporary state to fill the template
        temp_state = state.copy()
        temp_state.update(fill_data)
        
        # Fill the template (fill_template handles all placeholders)
        filled_prompt = fill_template(template_str, temp_state)
        
        # If there is planning_text (the CoT plan), add it to the prompt
        if planning_text:
            filled_prompt += f"\n\n====================\nImplementation Plan (CoT)\n====================\n{planning_text}\n\n**Important**: Based on the implementation plan above, write the actual code. The plan provides the strategy, but you must implement it correctly with proper Python syntax and handle all edge cases.\n"
        
        # If there is a guidance_message, add it to the end of the prompt
        if guidance_message:
            print(f"   Applying guidance from previous error fix...")
            print(f"     Guidance: {guidance_message[:200]}...")
            filled_prompt += f"\n\n====================\nPrevious Error Guidance\n====================\n{guidance_message}\n"
        
        # Call the LLM to generate code (with a retry mechanism)
        print(f"\n[Code Generation] Generating code for feature '{feature.get('name')}'...")
        print(f"  Prompt length: {len(filled_prompt)} characters")
        
        messages = [
            HumanMessage(content=filled_prompt)
        ]
        try:
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"  [ERROR] LLM API call failed: {type(e).__name__}: {str(e)}")
            raise
        
        # Extract the code
        cleaned_code = self._extract_function(response_text)
        
        print(f"  Response length: {len(response_text)} characters")
        if cleaned_code:
            print(f"  [OK] Code extracted successfully, code length: {len(cleaned_code)} characters")
        else:
            print(f"  [WARN]  Could not extract valid code from the response")
        
        return CodeResult(
            code=cleaned_code,
            prompt=filled_prompt,
            response=response_text,
            planning_response=planning_text if planning_text else None
        )
    
    def _extract_function(self, text: str) -> Optional[str]:
        """Extract the extract function from the LLM response
        
        Args:
            text: LLM response text
            
        Returns:
            The extracted code string, or None if not found
        """
        # Remove the code fence markers
        text = self._remove_code_fences(text)
        
        # Find def extract
        def_index = text.find("def extract")
        if def_index < 0:
            return None
        
        # Extract the code from def extract to the end of the file
        code = text[def_index:].strip()
        
        # Ensure the code ends with a newline
        if not code.endswith('\n'):
            code += '\n'
        
        return code
    
    def _remove_code_fences(self, text: str) -> str:
        """Remove the markdown code fence markers"""
        start = text.find("```")
        if start < 0:
            return text
        
        end = text.rfind("```")
        if end > start:
            text = text[start + 3:end]
        else:
            text = text[start + 3:]
        
        # Remove the language identifier (such as python)
        text = text.lstrip()
        if text.startswith("python"):
            text = text[6:].lstrip()
        elif text.startswith("py"):
            text = text[2:].lstrip()
        
        return text
